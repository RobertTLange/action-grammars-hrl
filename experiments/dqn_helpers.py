import argparse
import math
import numpy as np
from collections import deque

def command_line_dqn():
    parser = argparse.ArgumentParser()
    parser.add_argument('-roll_eps', '--ROLLOUT_EVERY', action="store",
                        default=10, type=int,
                        help='Rollout test performance after # batch updates.')
    parser.add_argument('-save_eps', '--SAVE_EVERY', action="store",
                        default=50, type=int,
                        help='Save network and learning stats after # batch updates')
    parser.add_argument('-update_eps', '--UPDATE_EVERY', action="store",
                        default=10, type=int,
                        help='Update target network after # batch updates')
    parser.add_argument('-n_eps', '--NUM_EPISODES', action="store",
                        default=100, type=int,
                        help='# Epochs to train for')
    parser.add_argument('-max_steps', '--MAX_STEPS', action="store",
                        default=1000, type=int,
                        help='Max # of steps before episode terminated')
    parser.add_argument('-v', '--verbose', action="store_true", default=False,
                        help='Get training progress printed out')


    parser.add_argument('-gamma', '--GAMMA', action="store",
                        default=0.9, type=float,
                        help='Discount factor')
    parser.add_argument('-l_r', '--L_RATE', action="store", default=0.001,
                        type=float, help='Save network and learning stats after # epochs')
    parser.add_argument('-e_start', '--EPS_START', action="store", default=1,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_start', '--EPS_STOP', action="store", default=0.01,
                        type=float, help='Start Exploration Rate')
    parser.add_argument('-e_start', '--EPS_DECAY', action="store", default=500,
                        type=float, help='Start Exploration Rate')



    parser.add_argument('-train_batch', '--TRAIN_BATCH_SIZE', action="store",
                        default=256, type=int, help='# images in training batch')
    parser.add_argument('-model', '--MODEL_TYPE', action="store",
                        default="architecture_1", type=str, help='FKP model')


    parser.add_argument('-device', '--device_id', action="store",
                        default=0, type=int, help='Device id on which to train')
    parser.add_argument('-chkp_path', '--checkpoint_path', action="store",
                        default="models/saved_facemark_models/architecture_1_2019-05-23_keypoints_model.pt", type=str, help='Path to store online agents params')
    return parser.parse_args()


class ReplayBuffer(object):
    def __init__(self, capacity, record_macros=False):
        self.buffer = deque(maxlen=capacity)
        self.record_macros = record_macros

    def push(self, ep_id, state, action,
             reward, next_state, done, macro=None):
        state = state
        next_state = next_state

        if self.record_macros:
            self.buffer.append((ep_id, state, action, macro,
                                reward, next_state, done))
        else:
            self.buffer.append((ep_id, state, action,
                                reward, next_state, done))

    def push_policy(self, ep_id, state, action, next_state):
        state = state
        next_state = next_state

        if self.record_macros:
            self.buffer.append((ep_id, state, action, macro, next_state))
        else:
            self.buffer.append((ep_id, state, action, next_state))

    def sample(self, batch_size):
        if not self.record_macros:
            ep_id, state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
            return ep_id, np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def epsilon_by_episode(eps_id, epsilon_start, epsilon_final, epsilon_decay):
    eps = (epsilon_final + (epsilon_start - epsilon_final)
           * math.exp(-1. * eps_id / epsilon_decay))
    return eps


def update_target(current_model, target_model):
    # Transfer parameters from current model to target model
    target_model.load_state_dict(current_model.state_dict())


def compute_td_loss(agent, optimizer, replay_buffer, args, Variable):
    obs, acts, rew, next_obs, done = replay_buffer.sample(args.batch_size, agent_id, params.reward_type["indiv_rewards"])

    # Flatten the visual fields into vectors for MLP - not needed for CNN!
    if params.agent_type == "MLP":
        agent_vf = [vf.flatten() for vf in agent_vf]
        agent_next_vf = [vf.flatten() for vf in agent_next_vf]

    agent_vf = Variable(torch.FloatTensor(np.float32(agent_vf)))
    agent_next_vf = Variable(torch.FloatTensor(np.float32(agent_next_vf)))
    action = Variable(torch.LongTensor(agent_actions))
    done = Variable(torch.FloatTensor(done))

    # Select either global aggregated reward if float or agent-specific if dict
    if type(reward[0]) == np.float64 or type(reward[0]) == int:
        reward = Variable(torch.FloatTensor(reward))
    elif type(reward[0]) == dict:
        rew_temp = [rew[agent_id] for rew in reward]
        reward = Variable(torch.FloatTensor(rew_temp))

    q_values = agents[agent_id]["current"](agent_vf)
    next_q_values = agents[agent_id]["target"](agent_next_vf)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + params.gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    # Perform optimization step for agent
    optimizers[agent_id].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(agents[agent_id]["current"].parameters(), 0.5)
    optimizers[agent_id].step()

    return loss


def get_logging_stats(env, agents, params):
    steps = []
    rewards = []

    for i in range(params.num_rollouts):
        step_temp, reward_temp, buffer = rollout_episode(env, agents, params)
        steps.append(step_temp)
        rewards.append(reward_temp)

    steps = np.array(steps)
    rewards = np.array(rewards)

    reward_stats = {
        "sum_mean": rewards_sum.mean(),
        "sum_sd": rewards_sum.std(),
        "sum_median": np.median(rewards_sum),
        "sum_10_percentile": np.percentile(rewards_sum, 10),
        "sum_90_percentile": np.percentile(rewards_sum, 90),

        "survival_mean": rewards_survival.mean(),
        "survival_sd": rewards_survival.std(),
        "survival_median": np.median(rewards_survival),
        "survival_10_percentile": np.percentile(rewards_survival, 10),
        "survival_90_percentile": np.percentile(rewards_survival, 90),

        "attraction_mean": rewards_attraction.mean(),
        "attraction_sd": rewards_attraction.std(),
        "attraction_median": np.median(rewards_attraction),
        "attraction_10_percentile": np.percentile(rewards_attraction, 10),
        "attraction_90_percentile": np.percentile(rewards_attraction, 90),

        "repulsion_mean": rewards_repulsion.mean(),
        "repulsion_sd": rewards_repulsion.std(),
        "repulsion_median": np.median(rewards_repulsion),
        "repulsion_10_percentile": np.percentile(rewards_repulsion, 10),
        "repulsion_90_percentile": np.percentile(rewards_repulsion, 90),

        "alignment_mean": rewards_alignment.mean(),
        "alignment_sd": rewards_alignment.std(),
        "alignment_median": np.median(rewards_alignment),
        "alignment_10_percentile": np.percentile(rewards_alignment, 10),
        "alignment_90_percentile": np.percentile(rewards_alignment, 90)
    }

    steps_stats = {
        "mean": steps.mean(),
        "sd": steps.std(),
        "median": np.median(steps),
        "10_percentile": np.percentile(steps, 10),
        "90_percentile": np.percentile(steps, 90)
    }
    return reward_stats, steps_stats, buffer
