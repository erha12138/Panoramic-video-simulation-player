import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 定义 DQN 网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
       
        # 假设输入是二维数组，这里直接使用输入维度
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        test_input = torch.randn(1, 1, *input_dim)
        test_output = self.pool(torch.relu(self.conv1(test_input)))
        test_output = self.pool(torch.relu(self.conv2(test_output)))
        flattened_size = test_output.view(1, -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x):
        # 确保输入维度为 4D，形状为 [batch_size, 1, 6, 8]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(1)

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)

        # 确保输出形状为 (1, 8)，如果是单样本输入
        if x.size(0) == 1:
            x = x.unsqueeze(0)

        return x


# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)

    def __len__(self):
        return len(self.buffer)

# 定义 DQN 代理 
class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=0.8, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64, buffer_capacity=10000):
        self.input_dim = (6,8)
        self.output_dim = 8
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.eval_model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.target_model.load_state_dict(self.eval_model.state_dict())
        self.optimizer = optim.Adam(self.eval_model.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def act(self, state):
        state = np.array(state).reshape(6, 8)
        if(np.random.rand()>self.epsilon):
            return random.randrange(self.output_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.eval_model(state)
        q_values = q_values.reshape(-1)
        action = torch.argmax(q_values, dim=0).item()
        return action

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.add(state, action, reward, next_state)

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        q_values = self.eval_model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_value = loss.item() 
         
        # 计算动作熵
        all_q_values = self.eval_model(states)
        probabilities = torch.softmax(all_q_values, dim=1)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1).mean().item()

        # 记录 q_values 和动作熵
        if not hasattr(self, 'q_value_history'):
            self.q_value_history = []
            self.entropy_history = []
        self.q_value_history.append(q_values.mean().item())
        self.entropy_history.append(entropy)

        # 判断模型是否收敛
        if len(self.q_value_history) > 100:  # 至少记录 100 个值
            q_value_std = np.std(self.q_value_history[-100:])
            entropy_std = np.std(self.entropy_history[-100:])
            if q_value_std < 0.01 and entropy_std < 0.01:
                print("模型可能已经收敛")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss_value
    
    def get_100_q_value_history(self):
        if len(self.q_value_history) > 100:
            return self.q_value_history[-100:]
        else:
            return None
    def get_100_entropy_history(self):
        if len(self.q_value_history) > 100:
            return self.entropy_history[-100:]
        else:
            return None
#更新目标网络的参数，替换为决策网络参数
    def update_target_model(self):
        self.target_model.load_state_dict(self.eval_model.state_dict())


if __name__ == "__main__":
    # 假设输入维度和输出维度
    input_dim = 10
    output_dim = 5

    agent = DQNAgent(input_dim, output_dim)

    # 模拟训练过程
    for episode in range(100):
        state = np.random.rand(input_dim)
        done = False
        while not done:
            action = agent.act(state)
            # 模拟环境交互
            next_state = np.random.rand(input_dim)
            reward = np.random.rand()
            done = np.random.choice([True, False], p=[0.1, 0.9])

            agent.remember(state, action, reward, next_state)
            agent.replay()

            state = next_state

        if episode % 10 == 0:
            agent.update_target_model()