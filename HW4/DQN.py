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


class replay_buffer():
    '''
    A deque storing trajectories
    '''

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.

        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish
        
        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.

        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''

    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 4  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        '''
        Forward the state to the neural network.
        
        Parameter:
            states: a batch size of states
        
        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000):
        """
        The agent learning how to control the action of the cart pole.
        
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.env = env
        self.n_actions = 2  # the number of actions
        self.count = 0  # recording the number of iterations

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.target_net = Net(self.n_actions)  # the target network

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def learn(self):
        '''
        - Implement the learning function.
        - Here are the hints to implement.
        
        Steps:
        -----
        1. Update target net by current net every 100 times. (we have done for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        -----
        
        Parameters:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        
        Returns:
            None (Don't need to return anything)
        '''
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        # Begin your code
        # 隨機取樣 batch_size 個 experience
        b_observations, b_actions, b_rewards, b_next_observations, b_done=self.buffer.sample(self.batch_size)
        # 計算現有 eval net 和 target net 得出 Q value 的落差
        b_observations=torch.tensor(np.array(b_observations),dtype=torch.float)
        b_next_observations=torch.tensor(np.array(b_next_observations),dtype=torch.float)
        b_rewards=np.array(b_rewards).reshape(-1,1)
        b_rewards=torch.tensor(np.array(b_rewards),dtype=torch.float)
        b_actions=np.array(b_actions).reshape(-1,1)
        b_actions=torch.tensor(np.array(b_actions),dtype=torch.long)
        q_eval = self.evaluate_net(b_observations).gather(1, b_actions) # 重新計算這些 experience 當下 eval net 所得出的 Q value
        q_next = self.target_net.forward(b_next_observations).detach() # detach 才不會訓練到 target net
        if self.count!=200:
            for i in range(self.batch_size):
                if b_done[i]:
                    q_next[i]=0
        # q_target = b_rewards + self.gamma * q_next.max(1)[0].view(self.batch_size, 1) # 計算這些 experience 當下 target net 所得出的 Q value
        q_target = b_rewards + self.gamma * q_next.max(1).values.unsqueeze(-1)
        # print(q_eval, q_next,b_actions, q_target)
        lo=nn.MSELoss()
        loss = lo(q_eval, q_target)
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # End your code

        # You can add some conditions to decide when to save your neural network model
        
    def choose_action(self, state):
        """
        - Implement the action-choosing function.
        - Choose the best action with given state and epsilon
        
        Parameters:
            self: the agent itself.
            state: the current state of the enviornment.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        
        Returns:
            action: the chosen action.
        """
        with torch.no_grad():
            # Begin your code
            # x = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            if random.uniform(0,1)<self.epsilon:
                action=np.random.randint(0,self.n_actions)
            else:
                # act_val=self.evaluate_net.forward(torch.FloatTensor(state)).squeeze(0).detach()
                act_val=self.evaluate_net.forward(torch.FloatTensor(state)).squeeze(0)
                action=int(torch.argmax(act_val).numpy())
            # End your code
        return action

    def check_max_Q(self):
        """
        - Implement the function calculating the max Q value of initial state(self.env.reset()).
        - Check the max Q value of initial state
        
        Parameter:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        
        Return:
            max_q: the max Q value of initial state(self.env.reset())
        """
        # Begin your code
        return np.max(self.evaluate_net.forward((torch.FloatTensor(self.env.reset()))).detach().numpy())
        # End your code


def train(env):
    """
    Train the agent on the given environment.
    
    Paramenters:
        env: the given environment.
    
    Returns:
        None (Don't need to return anything)
    """
    agent = Agent(env)
    episode = 1000
    rewards = []
    max_=np.max(total_rewards) if len(total_rewards) else 0
    for _ in tqdm(range(episode)):
        state = env.reset()
        count = 0
        while True:
            count += 1
            agent.count += 1
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.insert(state, int(action), reward,
                                next_state, int(done))
            if agent.count >= 1000:
                agent.learn()
            if done:
                rewards.append(count)
                if count>= max_:
                    max_=count
                    if os.path.exists("./Tables/DQN.pt") and agent.count>=1000:
                        agent_=Agent(env)
                        agent_.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
                        b_observations, b_actions, _, _, _=agent.buffer.sample(agent.batch_size)
                        b_actions=torch.LongTensor(np.array(b_actions).reshape(-1,1))
                        b_observations=torch.FloatTensor(np.array(b_observations))
                        with torch.no_grad():
                            Q = agent.target_net.forward(b_observations).gather(1, b_actions).detach()    
                            Q_= agent_.target_net.forward(b_observations).gather(1, b_actions).detach()
                        if  (Q_<Q).all():
                            torch.save(agent.target_net.state_dict(), "./Tables/DQN.pt")
                    else:
                        torch.save(agent.target_net.state_dict(), "./Tables/DQN.pt")    
                break
            state = next_state
    total_rewards.append(rewards)


def test(env):
    """
    Test the agent on the given environment.
    
    Paramenters:
        env: the given environment.
    
    Returns:
        None (Don't need to return anything)
    """
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    for _ in range(100):
        state = env.reset()
        count = 0
        while True:
            count += 1
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, _, done, _ = env.step(action)
            if done:
                rewards.append(count)
                break
            state = next_state
    print(f"reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")


def seed(seed=20):
    '''
    It is very IMPORTENT to set random seed for reproducibility of your result!
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    '''
    The main funtion
    '''
    # Please change to the assigned seed number in the Google sheet
    SEED = 4

    env = gym.make('CartPole-v0')
    seed(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)
        
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    # training section:
    for i in range(5):
        print(f"#{i + 1} training progress")
        train(env)
    # testing section:
    test(env)
    
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))

    env.close()
