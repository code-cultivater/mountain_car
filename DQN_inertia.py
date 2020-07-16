import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
from manager_gpu import  device
# Hyper Parameters
BATCH_SIZE = 3200
LR = 0.01                   # learning rate
EPSILON = 0.8              # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 200000
env = gym.make('MountainCar-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        #global EPSILON
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        #EPSILON *= 1.0001
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self,t):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        #print(self.eval_net(b_s))
        #print( torch.argmax(q_next, dim=1))
        #loss = self.loss_func(q_eval, q_target)+10*torch.nn.CrossEntropyLoss(self.eval_net(b_s), torch.argmax(q_next, dim=1))/(t+1)
        loss = self.loss_func(q_eval, q_target)+10*torch.nn.CrossEntropyLoss()(self.eval_net(b_s), torch.argmax(q_next, dim=1))/(1*t+1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
TrainTimes=100
train_rewards=[]
for iter in range(TrainTimes):
    dqn = DQN()
    t = 0
    episodes_rewards = []
    print('\nCollecting experience...')

    for i_episode in range(100):
        s = env.reset()
        ep_r = 0
        steps = 0
        since = time.time()
        while True:
            #env.render()
            a = dqn.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            # modify the reward
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2


            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:

                dqn.learn(t)


            steps += 1;

            if done or steps >= 20000:
                t += 1
                episodes_rewards.append(ep_r)
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
                break
            s = s_
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))  # 打印出来时间
    train_rewards.append(episodes_rewards)

np.savetxt("DQN_inert",train_rewards)