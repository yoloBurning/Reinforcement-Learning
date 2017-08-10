import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import gym

import numpy as np
import random
from collections import deque


BATCH_SIZE = 32
LR = 0.05						# learning rate
EPSILON = 0.9					# greedy policy
GAMMA = 0.9						# reward discount
TARGET_REPLAY_ITER = 100		# target update frequency
MEMORY_CAPACITY = 50

env = gym.make('FrozenLake-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = 16  # env.observation_space.shape[0]


class Net(nn.Module):

    def __init__(self,):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(N_STATES, 20)
        # self.fc1.weight.data.normal_(0, 0.1) 	# initialization
        # self.out = nn.Linear(20, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)		# initialization
        self.fc1 = nn.Linear(N_STATES, N_ACTIONS)
        self.fc1.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # x = self.fc1(x)
        # x = F.relu(x)
        # action_value = self.out(x)
        action_value = self.fc1(x)
        return action_value


class DQN(object):

    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()  # define 2 networks

        self.learn_step_counter = 0				# for target update
        self.memory_counter = 0					# for storing memory

        self.memory = np.zeros((
            MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        if EPSILON > np.random.uniform():		# greedy
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()[0, 0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        action = self.eval_net.forward(x)
        action = torch.max(action, 1)[1].data.numpy()[0, 0]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net update
        if self.learn_step_counter % TARGET_REPLAY_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(
            b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(
            b_memory[:, N_STATES + 1: N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        q_e = self.eval_net(b_s)
        #print("Before: ", q_e)
        q_eval = q_e.gather(1, b_a)
        #print("After: ", q_eval)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0]

        # print type(q_target)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
print dqn.eval_net

for i_episode in xrange(5000):
    s = env.reset()
    s = np.eye(N_STATES)[s]
    if i_episode == 0:
        print s
    while True:
        # env.render()
        a = dqn.choose_action(s)

        s_, r, done, info = env.step(a)

        s_ = np.eye(N_STATES)[s_]

        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / \
        # env.theta_threshold_radians - 0.5
        # r = r1 + r2

        dqn.store_transition(s, a, r, s_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        s = s_
        if done:
            break

        
    if i_episode % 100 == 0:
        total_reward = 0
        for i in xrange(10):
            state = env.reset()
            
            for t in xrange(500):
                # env.render()
                s = np.eye(N_STATES)[state]
                action = dqn.action(s)  # direct action for test
                state, reward, done, _ = env.step(action)
                total_reward += reward
                # s = np.eye(N_STATES)[state]
                if done:
                    break
                if t > 400:
                    break
        ave_reward = total_reward / 10
        # l.append(ave_reward)
        print 'episode: ', i_episode, 'Evaluation Average Reward:', ave_reward
        if ave_reward >= 200:
            break
