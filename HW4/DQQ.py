
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
import os
from tqdm import tqdm
total_rewards = []

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_values = self.out(x)
        return action_values

class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity):
        

        # memory buffer, 每一筆經驗是 (state + next state + reward + action)
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
        self.memory_counter = 0 # buffer 中幾筆經驗了
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
        self.eval_net = Net(n_states, n_actions, n_hidden)
        self.target_net = Net(n_states, n_actions, n_hidden)
        self.target_replace_iter = target_replace_iter # target net 多久 update 一次
        self.learn_step_counter = 0 # 現在學多久了
        # 其他訓練需要的
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self,state):
        x = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        if np.random.uniform() < self.epsilon :
            action = np.random.randint(0, self.n_actions)
        else:
            # eval net 預測 Q-value
            action_values = self.eval_net(x)
            # 選 Q-value 最大的 action
            action = torch.argmax(action_values).item()
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))

        # 存進 memory；舊 memory 可能會被覆蓋
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
    	# 從 buffer 中隨機挑選經驗，將經驗分成 state、action、reward、next state
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.tensor(b_memory[:, :self.n_states], dtype=torch.float)
        b_action = torch.tensor(b_memory[:, self.n_states:self.n_states+1], dtype=torch.long)
        b_reward = torch.tensor(b_memory[:, self.n_states+1:self.n_states+2], dtype=torch.float)
        b_next_state = torch.tensor(b_memory[:, -self.n_states:], dtype=torch.float)
        # print(b_state,b_action)
        # print(b_memory[:, self.n_states:self.n_states+1])
        # print(b_action)
        # print(7)
        # 計算 eval net 的 Q-value 和 target net 的 loss
        q_eval = self.eval_net(b_state).gather(1, b_action) # 經驗當時的 Q-value
        q_next = self.target_net(b_next_state).detach()
        q_target = b_reward + self.gamma * q_next.max(1).values.unsqueeze(-1) # 目標 Q-value
        loss = self.loss_func(q_eval, q_target)
        # print(b_reward.size(), q_target.size())
        # loss = nn.MSELoss(q_eval, q_target)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network 一陣子更新一次
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())


if __name__ == "__main__":
    env = gym.make('CartPole-v0')

    # Environment parameters
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    # Hyper parameters
    n_hidden = 50
    batch_size = 5
    lr = 0.01                 # learning rate
    epsilon = 0.1             # epsilon-greedy
    gamma = 0.9               # reward discount factor
    target_replace_iter = 100 # target network 更新間隔
    memory_capacity = 50
    n_episodes = 4000

    # 建立 DQN
    dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity)

    # 學習
    for i_episode in range(n_episodes):
        t = 0
        rewards = 0
        state = env.reset()
        while True:
            env.render()

            # 選擇 action
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)

            # 儲存 experience
            dqn.store_transition(state, action, reward, next_state)

            # 累積 reward
            rewards += reward

            # 有足夠 experience 後進行訓練
            if dqn.memory_counter > memory_capacity:
                dqn.learn()

            # 進入下一 state
            state = next_state

            if done:
                print('Episode finished after {} timesteps, total rewards {}'.format(t+1, rewards))
                break

            t += 1

    env.close()