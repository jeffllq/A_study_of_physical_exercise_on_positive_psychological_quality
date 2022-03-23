import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

# Input_cols = [ '年龄', '性别', '家庭所在地', '家庭经济情况', '学习成绩', '对待体育活动的态度',
#   '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平','类别']

Input_cols = [ '剧烈运动MET', '适度运动MET', '步行MET', '总MET', '身体活动水平']

# Output_cols = ['认知', '人际', '情感', '公正', '节制', '超越', '积极心理品质得分']
Output_cols = [ '积极心理品质得分']


# 输入输出的数据维度，这里都是1维
INPUT_FEATURE_DIM = 5
OUTPUT_FEATURE_DIM = 1
# 隐含层中神经元的个数
NEURON_NUM = 5
# 学习率，越大学的越快，但也容易造成不稳定，准确率上下波动的情况
LEARNING_RATE = 0.001
#训练epoch
EPOCH = 20000
#样本数据集
# PATH = r'data/聚类结果20220322_剔除异常数据.csv'
# PATH = r'data/聚类结果_只有运动.csv'
PATH = r'data/聚类结果_只有运动_剔除异常数据.csv'

plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False#用来正常显示负号


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, header=0)
        # self.df = self.df [self.df ['类别']==2]
        x_df = self.df[Input_cols]
        y_df = self.df[Output_cols]
        # 归一化处理
        x_df = (x_df-x_df.min())/(x_df.max() - x_df.min())
        y_df = y_df/5

        # print(x_df_normalize)
        self.x_data = torch.from_numpy(np.array(x_df)).to(torch.float32)
        self.y_data = torch.from_numpy(np.array(y_df)).to(torch.float32)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]


class BPNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(BPNN, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(n_input, n_hidden) # 隐藏层
        self.hidden_layer2 = torch.nn.Linear(n_hidden, n_hidden+2) # 隐藏层
        self.hidden_layer3 = torch.nn.Linear(n_hidden+2, n_hidden+4) # 隐藏层
        # self.hidden_layer4 = torch.nn.Linear(n_hidden, n_hidden) # 隐藏层
        self.output_layer = torch.nn.Linear(n_hidden+4, n_output) # 输出层（预测层）

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = torch.relu(x)
        x = self.hidden_layer2(x)
        x = torch.relu(x)
        x = self.hidden_layer3(x)
        x = torch.relu(x)
        # x = self.hidden_layer4(x)
        # x = torch.relu(x)

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

def train(net, train_dataloader, validate_dataloader, test_dataloader):
    # 这里也可以使用其它的优化方法
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_func = torch.nn.MSELoss()

    epcoh_list = []
    train_loss_list = []
    validate_loss_list = []
    test_loss_list = []

    # plt.ion()  # 开启interactive mode 成功的关键函数
    plt.ioff()
    # plt.figure(1)

    for i in range(EPOCH):
        epcoh_list.append(i+1)
        loss = 0; loss0 = 0

        for x_data, y_data in train_dataloader:
            prediction = net(x_data)
            # 第一个参数为预测值，第二个为真值
            loss = loss_func(prediction, y_data)
            # print(loss)
            # 开始优化步骤
            # 每次开始优化前将梯度置为0
            optimizer.zero_grad()
            # 误差反向传播
            loss.backward()
            # 按照最小loss优化参数
            optimizer.step()

        train_loss_list.append(loss.item())

        for x, y in validate_dataloader:
            prediction = net(x)
            loss0 = loss_func(prediction, y)
        validate_loss_list.append(loss0.item())

        for x, y in test_dataloader:
            prediction = net(x)
            loss0 = loss_func(prediction, y)
        test_loss_list.append(loss0.item())


        # # 可视化训练结果
        # if i%2 == 0:
        #
        #     # 清空上一次显示结果
        #     plt.clf()
        #     # 实时损失的曲线
        #     plt.plot(epcoh_list, train_loss_list, c='red', lw='2')
        #     title = str(i)+'epoch'
        #     plt.title(title)
        #     plt.text(0.5, -0.5, 'epoch=%d loss=%.4f' % (i, train_loss_list[-1]), fontdict={'size': 15, 'color': 'blue'})
        #     plt.draw()
        #     plt.pause(0.1)


    # plt.subplot(1, 2, 1)
    plt.plot(epcoh_list, train_loss_list, c='red', lw='2', label='train loss')
    plt.plot(epcoh_list, validate_loss_list, c='blue', lw='2', label='validate loss')
    # title = 'Loss curve'
    # plt.title(title)
    # plt.xlabel('epoch')
    # plt.ylabel('MSE Loss')
    # plt.legend(loc='upper right')
    # plt.ylim(0, 0.1)

    # plt.subplot(1, 2, 2)
    plt.plot(epcoh_list, test_loss_list, c='green', linestyle='--', lw='2', label='test loss')
    title = 'Loss curve'
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('MSE Loss')
    plt.legend(loc='upper right')
    plt.ylim(0, 0.1)

    plt.show()


    # print(net)
    # 保存整个网络
    torch.save(net, 'BP0.pkl')
    # 只保存网络中节点的参数
    torch.save(net.state_dict(), 'BP0_params.pkl')
    return

def weight_init(m):
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        print("初始化权重")
        m.weight.data.normal_(0, 1)
        m.bias.data.zero_()

def test(dataloader):
    net_retore = torch.load('BP0.pkl')
    for x_data, y_data in dataloader:
        print('y:' , round(y_data.item()*5, 2))
        prediction = net_retore(x_data)
        print('prediction:', round(prediction.item()*5, 2))


if __name__ == '__main__':
    BP = BPNN(n_input=INPUT_FEATURE_DIM, n_hidden=NEURON_NUM, n_output=OUTPUT_FEATURE_DIM) # 建立神经网络
    BP.apply(weight_init)
    # print(BP)
    # x_data, y_data = load_data(PATH)

    data = MyDataset(PATH)
    data_length = data.__len__()
    print('样本量：',data_length)

    train_size = round(data_length*0.8)
    validate_size = round(data_length*0.1)
    test_size = data_length-train_size-validate_size

    train_dataset, validate_dataset, test_dataset = \
        torch.utils.data.random_split(data, lengths =[train_size, validate_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=False, num_workers=0)
    validate_dataloader = DataLoader(validate_dataset, batch_size=50, shuffle=True, drop_last=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=0)

    # train(BP, train_dataset, validate_dataloader, test_dataloader)

    test(test_dataloader)

