from __future__ import division, print_function
import os
import math
import numpy as np
import cPickle
from pickle import dumps,loads
from cfg_grammar import *
from hmmlearn import hmm

from keras.layers import *
from keras import Input
from keras.models import Sequential, load_model
from keras.utils.vis_utils import plot_model
import keras.backend as K

base_dir = os.getcwd()
results_dir = base_dir + "/results/"
trace_dir = base_dir + "/utils/trace_files/"
traces_hmm = trace_dir + "action_hist_data.txt"

# trace_no, stp, action, discounted_return(rewards, gamma),
# cur_state[0], cur_state[1], next_state[0], next_state[1]


def train_grammars(env, state_list, model_type):

    data_train, lengths_train = state_list, [len(seq) for seq in state_list]
    no_obs = env.observation_space.n

    if model_type[0] == "hmm":
        data_train = [[j] for i in data_train for j in i]
        startprob, transmat, emissionprob = init_hmm(no_obs, n_states=model_type[1])
        scfg_model = run_hmm(data_train, lengths_train, startprob, transmat, emissionprob)
    elif model_type[0] == "rnn":
        if model_type[1] == "lstm":
            scfg_model = lstmModel(no_obs, type=model_type[2])
        elif model_type[1] == "gru":
            scfg_model = gruModel(no_obs, type=model_type[2])

        sequences_train, unique_lengths_train = genSequences(data_train, lengths_train)
        for i in unique_lengths_train:
            # start = time.time()
            seq_in, seq_out = selectSequences(sequences_train, i)
            seq_in_enc, seq_out_enc = encode_data(no_obs, seq_in, seq_out, True)
            bs = len(seq_out)
            scfg_model.fit(seq_in_enc, seq_out_enc, epochs=5, batch_size=64, verbose=0)
            # print("Done fitting model for all {}-length sequences (Total: {}) after {} seconds".format(i, seq_in_enc.shape[0], time.time()-start))
    return scfg_model


def genSequences(data, lengths):
    """
    Input:
    Output:
    """
    sequences = []
    counter = 0

    for i in range(len(lengths)):
        len_seq = lengths[i]
        seq_temp = []
        for j in range(len_seq):
            seq_temp.append([data[i][j]])
            counter += 1
            if len(seq_temp)>=2:
                sequences.append(seq_temp[:])
    sequences.sort(key=len)

    return sequences, list(set(lengths))


def selectSequences(sequences, num_steps):
    """
    Input:
    Output:
    """
    seq_w_num_steps = []
    for h in range(len(sequences)):
        if len(sequences[h]) == num_steps:
            seq_w_num_steps.append(sequences[h])

    seq_all = np.array(seq_w_num_steps)
    seq_all = seq_all.reshape(seq_all.shape[0], seq_all.shape[1])
    seq_in, seq_out = seq_all[:, 0:(seq_all.shape[1]-1)], seq_all[:,-1]
    seq_out = seq_out.reshape(seq_out.shape[0], 1)
    # seq_in = seq_in.reshape(seq_in.shape[0], seq_in.shape[1], 1)
    return seq_in, seq_out


def encode_data(no_obs, seq_in, seq_out=None, in_and_out=False):
    """
    Input:
    Output:
    """
    model = Sequential()
    model.add(Embedding(input_dim=no_obs, output_dim=no_obs, input_length=None, embeddings_initializer='identity'))
    layer0_output_fcn = K.function([model.layers[0].input], [model.layers[0].output])
    seq_in_enc = layer0_output_fcn([seq_in])[0]
    if in_and_out:
        seq_out_enc = layer0_output_fcn([seq_out])[0]
        return seq_in_enc, seq_out_enc
    else:
        return seq_in_enc



class successor_buffer():
    def __init__(self, num_primitives):
        self.look_up = dict()
        self.num_primitives = num_primitives

    def get_valid_succesor(self, state, env):
        while type(state) == list:
            state = state[0]
        if state in self.look_up.keys():
            return self.look_up[state], range(self.num_primitives)
        else:
            return self.add_to_buffer(state, env)

    def add_to_buffer(self, state, env):
        valid_successors = []
        target_move = []
        snapshot = copy.deepcopy(env)

        for action in range(self.num_primitives):
            env = copy.deepcopy(snapshot)
            obs, r, done, info = env.step(action)
            valid_successors.append(obs)
            target_move.append(action)

        self.look_up[state] = valid_successors
        return valid_successors, range(self.num_primitives)


#######################################################################
# Model setup - Discrete Hidden Markov Model with Uniform Init
#######################################################################
def init_hmm(no_obs, n_states):
    """
    Input: number of desired hidden states for HMM model
    Output: Uniformly initialized matrices for HMM training
    """
    startprob = np.repeat(1./n_states, n_states)
    transmat = np.repeat(startprob, n_states, axis=0)
    temp = np.repeat(1./no_obs, no_obs)
    emissionprob = np.repeat([temp], n_states, axis=0)

    return startprob, transmat, emissionprob


def run_hmm(data_encoded, lengths, startprob, transmat, emissionprob,
            get_ic=False):
    """
    Input: Unique state id transformed data, length per ep trace and HMM inits
    Output: The trained HMM model with all attributes
    - Option: If desired get AIC and BIC for model as well
    """
    n_states = len(startprob)
    # Uniform Initialization
    model = hmm.MultinomialHMM(n_components=n_states)
    model.startprob = startprob
    model.transmat = transmat
    model.emissionprob = emissionprob

    model = model.fit(data_encoded, lengths)

    if get_ic:
        logprob, posteriors = model.score_samples(data_encoded, lengths)

        k = emissionprob.shape[1]
        p = n_states**2 + k*n_states - 1
        T = len(data_encoded)

        aic = -2*logprob + 2*p
        bic = -2*logprob + p*math.log(T)

        return model, aic, bic
    return model


def run_model_comparison(observations, data_raw, max_hidden):
    """
    Input: state encoding, raw data, max number of hidden steps, number folds for replay surprisal computation
    Output: Array of AICs, BICs and k-fold replay surprisal for different HMMs
    """
    AICs = []
    BICs = []
    surprisals = []

    data_full, lengths_full = string_to_code(data_raw, observations)

    hidden_seq = range(2, max_hidden + 1)
    for hiddens in hidden_seq:
        startprob, transmat, emissionprob = init_hmm(n_states=hiddens)

        model = run_hmm(data_full, lengths_full,
                        startprob, transmat, emissionprob)

        model, aic, bic = run_hmm(data_full, lengths_full,
                                  startprob, transmat, emissionprob,
                                  True)

        print("HMM for {} hidden states - AIC: {}".format(hiddens, aic))
        print("HMM for {} hidden states - BIC: {}".format(hiddens, bic))

        AICs.append(aic)
        BICs.append(bic)

    min_aic = AICs.index(min(AICs)) + 2
    min_bic = BICs.index(min(BICs)) + 2

    print("---------------------------------------------------------")
    print("Optimal Number of Hidden States - AIC {}".format(min_aic))
    print("Optimal Number of Hidden States - BIC {}".format(min_bic))
    return np.array([AICs, BICs])


def lstmModel(no_obs, type):
    # Model definition - Careful: Return sequences returns all hidden states
    # Pred output shape has no_timesteps as input but we only need final!
    model = Sequential()
    if type == 1:
        model.add(LSTM(5, batch_input_shape=(None, None, no_obs), return_sequences=True, stateful=False))
        model.add(Dense(no_obs, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    if type == 2:
        model.add(LSTM(10, batch_input_shape=(None, None, no_obs), return_sequences=True, stateful=False))
        model.add(Dense(no_obs, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    if type == 3:
        model.add(LSTM(15, batch_input_shape=(None, None, no_obs), return_sequences=True, stateful=False))
        model.add(Dense(no_obs, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def gruModel(no_obs, type):
    # Model definition - Careful: Return sequences returns all hidden states
    # Pred output shape has no_timesteps as input but we only need final!
    model = Sequential()
    if type == 1:
        model.add(GRU(5, batch_input_shape=(None, None, no_obs), return_sequences=True, stateful=False))
        model.add(Dense(no_obs, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    if type == 2:
        model.add(GRU(10, batch_input_shape=(None, None, no_obs), return_sequences=True, stateful=False))
        model.add(Dense(no_obs, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    if type == 3:
        model.add(GRU(15, batch_input_shape=(None, None, no_obs), return_sequences=True, stateful=False))
        model.add(Dense(no_obs, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#######################################################################
# Reinforcement Learning Fct - Action Selection and Following Option
#######################################################################

def hmm_action_selection(env, model, state_seq, successor_memory, sample=False):
    """
    Input: HMM model and a state_seq
    Output: Computes p(x_t|x_t-1,...,x_0)
    TODO: Make option to greedly select next action!
    """
    cur_state = state_seq[-1]
    valid_successors, moves = successor_memory.get_valid_succesor(cur_state, env)
    current_v_state, lengths = [state for state in state_seq], [len(state_seq)]
    # One HMM State Transition
    logprob, posteriors = model.score_samples(current_v_state, lengths)
    cond_distr = np.matmul(model.emissionprob_.T,
                           np.matmul(model.transmat_.T, posteriors.T))

    full_distr_next_v_state = cond_distr[:, -1].ravel()
    distr_next_v_state = full_distr_next_v_state[valid_successors].ravel()/sum(full_distr_next_v_state[valid_successors]).ravel()
    n_valid_v_states = len(distr_next_v_state)

    # Sample or argmax
    if sample:
        next_move = np.random.choice(n_valid_v_states, 1, p=distr_next_v_state)[0]
    else:
        next_move = np.argmax(distr_next_v_state)

    # next_v_state = valid_successors[0][next_move]
    return next_move, valid_successors, distr_next_v_state


def encode_fast(env, seq_enc):
    no_obs = env.observation_space.n
    length_seq = len(seq_enc)
    temp = np.zeros([length_seq, no_obs])
    for i in range(length_seq):
        temp[i, seq_enc[i]] = 1
    return temp.reshape(1, length_seq, no_obs)


def rnn_action_selection(env, model, state_seq, successor_memory, sample=False):
    """
    Input: LSTM model and a state_seq
    Output: Computes p(x_t|x_t-1,...,x_0)
    TODO: Make option to greedly select next action! - argmax of softmax
    """
    valid_successors, moves = successor_memory.get_valid_succesor(state_seq[-1], env)
    seq_in = encode_fast(env, state_seq)

    preds = model.predict(seq_in)
    softmax_out = preds[0, -1, :]

    full_distr_next_v_state = softmax_out.ravel()
    distr_next_v_state = full_distr_next_v_state[valid_successors]/sum(full_distr_next_v_state[valid_successors])
    n_valid_v_states = len(distr_next_v_state)

    # Sample or argmax
    if sample:
        next_move = np.random.choice(n_valid_v_states, 1, p=distr_next_v_state)[0]
    else:
        next_move = np.argmax(distr_next_v_state)

    # next_v_state = valid_successors[0][next_move]

    return next_move, valid_successors, distr_next_v_state


def option_step(env, cur_state, model, max_surprisal,
                max_scfg_steps, model_type, successor_memory):

    state_seq = [[cur_state]]
    stp = 0

    done = False
    alt_done = False

    rewards = []
    actions = []
    while not done and not alt_done:
        if model_type == "hmm":
            action, valid_successors, distr_next_v_state = hmm_action_selection(env, model, state_seq, successor_memory)
        elif model_type == "rnn":
            action, valid_successors, distr_next_v_state = rnn_action_selection(env, model, state_seq, successor_memory)

        next_state, reward, done, _ = env.step(action)

        stp += 1
        actions.append(action)
        rewards.append(reward)

        state_seq.append([next_state])
        surprisal_temp = -np.log2(distr_next_v_state[valid_successors.index(state_seq[-1][0])])

        if surprisal_temp > max_surprisal or stp > max_scfg_steps:
            alt_done = True

        cur_state = next_state
    # return actions, state_seq, rewards
    return state_seq, rewards, done, actions
