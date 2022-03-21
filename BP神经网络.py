import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

# Input_cols = [ '年龄', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度',
#   '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平','类别']

Input_cols = [ '年龄', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度',
  '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平']

# Output_cols = ['认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分']
Output_cols = [ '积极心理品质得分']


# 输入输出的数据维度，这里都是1维
INPUT_FEATURE_DIM = 11
OUTPUT_FEATURE_DIM = 1
# 隐含层中神经元的个数
NEURON_NUM = 9
# 学习率，越大学的越快，但也容易造成不稳定，准确率上下波动的情况
LEARNING_RATE = 0.1
#训练epoch
EPOCH = 100
#样本数据集
PATH = r'data/清洗后的研究数据20220220.csv'

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, header=0)
        x_df = self.df[Input_cols]
        y_df = self.df[Output_cols]
        self.x_data = torch.from_numpy(np.array(x_df)).to(torch.float32)
        self.y_data = torch.from_numpy(np.array(y_df/5)).to(torch.float32) #归一化，将得分限制在[0-1]的区间

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]


class BPNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(BPNN, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_input, n_hidden) # 隐藏层
        self.output_layer = torch.nn.Linear(n_hidden, n_output) # 输出层（预测层）

    def forward(self, x):
        x = self.hidden_layer(x)
        x = F.sigmoid(x)
        x = self.output_layer(x)
        # print('forward：', x)
        return x

def load_data(path):
    df = pd.read_csv(path, header=0)

    df_x_data = df[Input_cols]
    x_data = np.array(df_x_data)
    x_data = torch.tensor(x_data).to(torch.float32)

    df_y_data = df[Output_cols]
    y_data = np.array(df_y_data)
    y_data = torch.tensor(y_data).to(torch.float32)

    print(x_data)
    print(y_data)
    return x_data, y_data

def train(net, dataloader):
    # 这里也可以使用其它的优化方法
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_func = torch.nn.MSELoss()

    epcoh_list = []
    loss_list = []
    # plt.ion()  # 开启interactive mode 成功的关键函数
    plt.ioff()
    plt.figure(1)

    for i in range(EPOCH):
        epcoh_list.append(i+1)
        loss = 0

        for x_data, y_data in dataloader:
            # print('X:', data_value, 'Y:', target_value)
            # 输入数据进行预测
            prediction = net(x_data)
            # print('预测值: ', 5*prediction)
            # 计算预测值与真值误差，注意参数顺序问题
            # 第一个参数为预测值，第二个为真值
            loss = loss_func(prediction, y_data)
            # print('损值: ', loss)
            # print('转换: ', loss.item())
            # 开始优化步骤
            # 每次开始优化前将梯度置为0
            optimizer.zero_grad()
            # 误差反向传播
            loss.backward()
            # 按照最小loss优化参数
            optimizer.step()
        loss_list.append(loss.item())

        # 可视化训练结果
        if i%2 == 0:
            print(len(loss_list))
            # 清空上一次显示结果
            plt.clf()
            # 实时损失的曲线
            plt.plot(epcoh_list, loss_list, c='red', lw='2')
            title = '第'+str(len(loss_list))+'次epoch'
            plt.title(title)
            plt.text(-0.5, -65, 'epoch=%d loss=%.4f' % (i, loss_list[-1]), fontdict={'size': 15, 'color': 'red'})
            plt.draw()
            plt.pause(0.1)

    plt.plot(epcoh_list, loss_list, c='red', lw='2')
    title = 'final loss curve'
    plt.title(title)
    plt.show()


    # print(net)
    # 保存整个网络
    torch.save(net, 'BP.pkl')
    # 只保存网络中节点的参数
    torch.save(net.state_dict(), 'BP_params.pkl')
    return

def test(dataloader):
    net_retore = torch.load('BP.pkl')
    for x_data, y_data in dataloader:
        print('y:' , 5*y_data)
        prediction = 5*net_retore(x_data)
        print('prediction:', prediction)


if __name__ == '__main__':
    BP = BPNN(n_input=INPUT_FEATURE_DIM, n_hidden=NEURON_NUM, n_output=OUTPUT_FEATURE_DIM) # 建立神经网络

    # print(BP)
    # x_data, y_data = load_data(PATH)

    data = MyDataset(PATH)
    data_length = data.__len__()
    train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(data, lengths =[300, 100, 48])
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=0)

    train(BP, train_dataset)
    # test(test_dataloader)

