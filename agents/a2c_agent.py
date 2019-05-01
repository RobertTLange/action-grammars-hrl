import gym
import gym_hanoi

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(N_INPUTS, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)

        self.actor = nn.Linear(64, N_ACTIONS)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        x = F.relu(x)

        return x

    def get_action_probs(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x))
        return action_probs

    def evaluate_actions(self, x):
        x = self(x)
        action_probs = F.softmax(self.actor(x))
        state_values = self.critic(x)

        return action_probs, state_values


def test_agent(agent):
    steps = 0
    done = False
    env = gym.make("Hanoi-v0")
    state = env.reset()
    global action_probs
    while not done:
        steps += 1
        s = torch.from_numpy(state).float().unsqueeze(0)

        action_probs = agent.get_action_probs(Variable(s))
        # Greedy action execution
        _, action_index = action_probs.max(1)
        action = action_index.data[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state

    return steps


def train_a2c_agent():
    scores = []
    num_games = []
    value_losses = []
    action_gains = []

    for i in range(N_GAMES):

        del states[:]
        del actions[:]
        del rewards[:]

        state = env.reset()
        done = False

        # act phase
        while not done:
            s = torch.from_numpy(state).float().unsqueeze(0)

            action_probs = model.get_action_probs(Variable(s))
            action = action_probs.multinomial().data[0][0]
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        if True:
            R = []
            rr = rewards
            rr.reverse()

            next_return = -1

            for r in range(len(rr)):
                this_return = rr[r] + next_return * .9
                R.append(this_return)
                next_return = this_return
            R.reverse()

            rewards = R

            # taking only the last 20 states before failure. wow this really improves training
            """rewards = rewards[-20:]
            states = states[-20:]
            actions = actions[-20:]"""

            global ss
            ss = Variable(torch.FloatTensor(states))

            global next_states
            action_probs, state_values, next_states = model.evaluate_actions(ss)

            next_state_pred_loss = (ss[1:] - next_states[:-1]).pow(2).mean()

            action_log_probs = action_probs.log()
            advantages = Variable(torch.FloatTensor(rewards)).unsqueeze(1) - state_values
            a = Variable(torch.LongTensor(actions).view(-1,1))
            chosen_action_log_probs = action_log_probs.gather(1, a)
            action_gain = (chosen_action_log_probs * advantages).mean()
            value_loss = advantages.pow(2).mean()
            total_loss = value_loss/50.0 - action_gain


            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()



        else: print("Not training, score of ", len(rewards))

        if i % 20 == 0:
            s = test_model(model)
            scores.append(s)
            num_games.append(i)

            action_gains.append(action_gain.data.numpy()[0])
            value_losses.append(value_loss.data.numpy()[0])
