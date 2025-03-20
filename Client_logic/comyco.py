import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import warnings
import pool

RAND_RANGE = 1000
FEATURE_NUM = 128
A_DIM = 8
# you can use mish active function instead
# the total performance will be improved a little bit.
def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

# 将 Net 类定义移到全局作用域
class Net(nn.Module):
    def __init__(self, S_INFO, S_LEN, A_DIM):
        super(Net, self).__init__()
        # 这里是 Net 类的初始化代码
        self.A_DIM=A_DIM
        self.fc0 = nn.Linear(S_LEN, FEATURE_NUM)
        self.fc1 = nn.Linear(S_LEN, FEATURE_NUM)
        self.conv2 = nn.Conv1d(1, 160, kernel_size=8, stride=1, padding=0)
        self.conv3 = nn.Conv1d(1, 160, kernel_size=8, stride=1, padding=0)
        self.conv4 = nn.Conv1d(1, 160, kernel_size=8, stride=1, padding=0)
        self.conv5 = nn.Conv1d(1, 160, kernel_size=8, stride=1, padding=0)
        self.fc6 = nn.Linear(1, FEATURE_NUM)
        self.gru = nn.GRU(7 * FEATURE_NUM, FEATURE_NUM)
        self.fc_out = nn.Linear(FEATURE_NUM, A_DIM)
        pass

    def forward(self, inputs):
        # 这里是前向传播代码
        # 调整输入检查，适应新的输入维度
        if inputs.shape != (1, 6, 8):
            assert inputs.shape[1:] == (1, 6, 8), f"Input shape should be (batch_size, 1, 6, 8), but got {inputs.shape}"
        # 去除多余的维度，因为这里不需要额外的维度 1
        inputs = inputs.squeeze(1)

        split_0 = torch.relu(self.fc0(inputs[:, 0, :]))
        split_1 = torch.relu(self.fc1(inputs[:, 1, :]))

        split_2 = torch.relu(self.conv2(inputs[:, 2:3, :]))
        split_3 = torch.relu(self.conv3(inputs[:, 3:4, :]))
        split_4 = torch.relu(self.conv4(inputs[:, 4:5, :]))
        split_5 = torch.relu(self.conv5(inputs[:, 5:6, :]))

        split_2_flat = split_2.view(split_2.size(0), -1)
        split_3_flat = split_3.view(split_3.size(0), -1)
        split_4_flat = split_4.view(split_4.size(0), -1)
        split_5_flat = split_5.view(split_5.size(0), -1)

        merge_net = torch.cat([split_0, split_1, split_2_flat,
                                split_3_flat, split_4_flat, split_5_flat], dim=1)
        # 为了后续 GRU 层输入形状要求，增加一个维度
        merge_net = merge_net.unsqueeze(1)

        dense_net_0, _ = self.gru(merge_net)

        out = torch.softmax(self.fc_out(dense_net_0[:, -1, :]), dim=1)

        assert out.shape[1:] == (8,), f"Output shape should be (batch_size, 8), but got {out.shape}"

        return out
# return Net(self.S_INFO, self.S_LEN, self.A_DIM)

def create_network(S_INFO, S_LEN, A_DIM):
    # 直接使用全局定义的 Net 类
    net = Net(S_INFO, S_LEN, A_DIM)
    return net

class libcomyco:
    def __init__(self, S_INFO, S_LEN, A_DIM,LR_RATE=1e-4,ID=1):
        super(libcomyco, self).__init__()  # 调用父类的初始化方法
        self.pool_ = pool.pool()
        self.S_INFO = S_INFO
        self.S_LEN = S_LEN
        self.A_DIM = A_DIM
        self.s_name = 'actor/' + str(ID)
        # self.net = self.create_network()
        self.criterion = nn.CrossEntropyLoss()
        self.net = create_network(S_INFO, S_LEN, A_DIM)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR_RATE)
        

    def predict(self, state):
        # 预测代码
        state = torch.tensor(state, dtype=torch.float32)
        action_prob = self.net(state).detach().numpy()
        action_cumsum = np.cumsum(action_prob)
        bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
        return action_prob, bit_rate


    def submit(self, state, action_real_vec):
        # 提交代码
        self.pool_.submit(state, action_real_vec)

    def train(self):
        # 训练代码
        training_s_batch, training_a_batch = self.pool_.get()
        if training_s_batch.shape[0] > 0:
            training_s_batch = torch.tensor(training_s_batch, dtype=torch.float32)
            training_a_batch = torch.tensor(training_a_batch, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self.net(training_s_batch)
            loss_ = self.criterion(output, training_a_batch) + 1e-3 * torch.sum(output * torch.log(output))
            loss_.backward()
            self.optimizer.step()

    def loss(self, state, action_real_vec):
        # 损失计算代码
        state = torch.tensor(state, dtype=torch.float32)
        action_real_vec = torch.tensor(action_real_vec, dtype=torch.float32).unsqueeze(0)
        output = self.net(state)
        loss_ = self.criterion(output, action_real_vec) + 1e-3 * torch.sum(output * torch.log(output))
        return loss_.item()
        

    def compute_entropy(self, x):
        # 熵计算代码
        """
        Given vector x, computes the entropy
        H(x) = - sum( p * log(p))
        """
        H = 0.0
        x = np.clip(x, 1e-5, 1.)
        for i in range(len(x)):
            if 0 < x[i] < 1:
                H -= x[i] * np.log(x[i])
        return H
   
# class libcomyco(nn.Module):
#     def __init__(self, S_INFO, S_LEN, A_DIM, LR_RATE=1e-4, ID=1):
#         super(libcomyco, self).__init__()
#         self.pool_ = pool.pool()
#         self.S_INFO = S_INFO
#         self.S_LEN = S_LEN
#         self.A_DIM = A_DIM
#         self.s_name = 'actor/' + str(ID)

#         self.net = self.create_network()
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.parameters(), lr=LR_RATE)
#         self.saver = None  # PyTorch doesn't have an exact equivalent to tf.train.Saver

    # def create_network(self):
    #     class Net(nn.Module):
    #         def __init__(self, S_INFO, S_LEN, A_DIM):
    #             super(Net, self).__init__()
    #             self.A_DIM=A_DIM
    #             self.fc0 = nn.Linear(S_LEN, FEATURE_NUM)
    #             self.fc1 = nn.Linear(S_LEN, FEATURE_NUM)
    #             self.conv2 = nn.Conv1d(1, 160, kernel_size=8, stride=1, padding=0)
    #             self.conv3 = nn.Conv1d(1, 160, kernel_size=8, stride=1, padding=0)
    #             self.conv4 = nn.Conv1d(1, 160, kernel_size=8, stride=1, padding=0)
    #             self.conv5 = nn.Conv1d(1, 160, kernel_size=8, stride=1, padding=0)
    #             self.fc6 = nn.Linear(1, FEATURE_NUM)
    #             self.gru = nn.GRU(7 * FEATURE_NUM, FEATURE_NUM)
    #             self.fc_out = nn.Linear(FEATURE_NUM, A_DIM)

    #         def forward(self, inputs):
    #             # 调整输入检查，适应新的输入维度
    #             if inputs.shape != (1, 6, 8):
    #                 assert inputs.shape[1:] == (1, 6, 8), f"Input shape should be (batch_size, 1, 6, 8), but got {inputs.shape}"
    #             # 去除多余的维度，因为这里不需要额外的维度 1
    #             inputs = inputs.squeeze(1)

    #             split_0 = torch.relu(self.fc0(inputs[:, 0, :]))
    #             split_1 = torch.relu(self.fc1(inputs[:, 1, :]))

    #             split_2 = torch.relu(self.conv2(inputs[:, 2:3, :]))
    #             split_3 = torch.relu(self.conv3(inputs[:, 3:4, :]))
    #             split_4 = torch.relu(self.conv4(inputs[:, 4:5, :]))
    #             split_5 = torch.relu(self.conv5(inputs[:, 5:6, :]))

    #             split_2_flat = split_2.view(split_2.size(0), -1)
    #             split_3_flat = split_3.view(split_3.size(0), -1)
    #             split_4_flat = split_4.view(split_4.size(0), -1)
    #             split_5_flat = split_5.view(split_5.size(0), -1)

    #             merge_net = torch.cat([split_0, split_1, split_2_flat,
    #                                    split_3_flat, split_4_flat, split_5_flat], dim=1)
    #             # 为了后续 GRU 层输入形状要求，增加一个维度
    #             merge_net = merge_net.unsqueeze(1)

    #             dense_net_0, _ = self.gru(merge_net)

    #             out = torch.softmax(self.fc_out(dense_net_0[:, -1, :]), dim=1)

    #             assert out.shape[1:] == (8,), f"Output shape should be (batch_size, 8), but got {out.shape}"

    #             return out

    #     return Net(self.S_INFO, self.S_LEN, self.A_DIM)

    # def predict(self, state):
    #     state = torch.tensor(state, dtype=torch.float32)
    #     action_prob = self.net(state).detach().numpy()
    #     action_cumsum = np.cumsum(action_prob)
    #     bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
    #     return action_prob, bit_rate

    # def loss(self, state, action_real_vec):
    #     state = torch.tensor(state, dtype=torch.float32)
    #     action_real_vec = torch.tensor(action_real_vec, dtype=torch.float32).unsqueeze(0)
    #     output = self.net(state)
    #     loss_ = self.criterion(output, action_real_vec) + 1e-3 * torch.sum(output * torch.log(output))
    #     return loss_.item()

    # def submit(self, state, action_real_vec):
    #     self.pool_.submit(state, action_real_vec)

    # def train(self):
    #     training_s_batch, training_a_batch = self.pool_.get()
    #     if training_s_batch.shape[0] > 0:
    #         training_s_batch = torch.tensor(training_s_batch, dtype=torch.float32)
    #         training_a_batch = torch.tensor(training_a_batch, dtype=torch.float32)
    #         self.optimizer.zero_grad()
    #         output = self.net(training_s_batch)
    #         loss_ = self.criterion(output, training_a_batch) + 1e-3 * torch.sum(output * torch.log(output))
    #         loss_.backward()
    #         self.optimizer.step()


    # def load(self, filename):
    #     self.load_state_dict(torch.load(filename))

    # def compute_entropy(self, x):
    #     """
    #     Given vector x, computes the entropy
    #     H(x) = - sum( p * log(p))
    #     """
    #     H = 0.0
    #     x = np.clip(x, 1e-5, 1.)
    #     for i in range(len(x)):
    #         if 0 < x[i] < 1:
    #             H -= x[i] * np.log(x[i])
    #     return H