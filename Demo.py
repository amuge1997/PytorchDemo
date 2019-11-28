import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch,numpy as np

# 准确度度量
def accuracy(Y,L):
    Ysize = X.size(0)
    Lsize = L.size(0)
    if Ysize != Lsize:
        print('accuracy()  Y,L 的维度不一致')
        return False
    # 标签转换为 int 类型
    L = L.int()

    correct = 0
    number = Ysize
    for y,l in zip(Y,L):
        # 以0.5为阈值归为0或者1
        y = y.ge(0.5).int()
        if False not in torch.eq(y,l):
            correct += 1
    return correct/number

class Model(nn.Module):  # 定义模型
    def __init__(self):
        nn.Module.__init__(self)
        # 输入维度为2,输出维度为2
        self.linear1 = nn.Linear(2, 2)
    def forward(self, X):
        Y = torch.softmax(self.linear1(X), dim=1)
        return Y

# 实例化模型,模型中保存着模型参数
model = Model()

# 实例化optimizer对象,能够保持模型当前的参数状态,
# 并基于计算得到的梯度进行模型参数更新
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 实例化损失函数,BECLoss二分类交叉熵
lossFunc = nn.BCELoss()

def train(X,L,Xt,Lt):
    for epoch in range(100):
        # 模型转换为训练模式
        model.train()
        # 与自动梯度有关
        Xval = Variable(X)
        Lval = Variable(L)
        # 正向计算获得输出
        Y = model(Xval)
        # 与模型连接
        loss = lossFunc(Y, Lval)
        # 梯度初始化归零,准备优化
        optimizer.zero_grad()
        # 反向传播,更新梯度
        loss.backward()
        # 根据计算得到的梯度,结合优化器参数进行模型参数更新
        optimizer.step()
        # 获取损失值
        train_loss = loss.item()
        # 准确度
        train_acc = accuracy(Y, L)

        print('训练集\n    偏差为:{}\n    准确度:{}%'.format(train_loss,train_acc*100))

        # 模型转换为评估模式
        model.eval()
        # 与自动梯度有关
        Xtval = Variable(Xt)
        Ltval = Variable(Lt)
        # 正向计算获得输出
        Yt = model(Xtval)
        # 计算损失
        tloss = lossFunc(Yt, Ltval)
        # 获取损失值
        test_loss = tloss.item()
        # 准确度
        test_acc = accuracy(Yt, Lt)
        print('测试集\n    偏差为:{}\n    准确度:{}%'.format(test_loss, test_acc*100))

        print()


if __name__ == '__main__':
    Xn = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ], dtype=np.float32)
    Ln = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
    ], dtype=np.float32)

    X = torch.from_numpy(Xn)  # 训练样本,假如X的shape为m*n,m个样本,每个样本n个维度
    L = torch.from_numpy(Ln)  # 训练标签,同理

    Xn = np.array([
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 1],
    ], dtype=np.float32)
    Ln = np.array([
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
    ], dtype=np.float32)
    Xt = torch.from_numpy(Xn)  # 测试样本,同理
    Lt = torch.from_numpy(Ln)  # 测试标签,同理

    train(X,L,Xt,Lt)




