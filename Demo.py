import torch.nn as nn
import torch.optim as optim
import torch,numpy as np
import matplotlib.pyplot as p

# 准确度度量
def accuracy(Y,L):
    Ysize = Y.size(0)
    Lsize = L.size(0)
    if Ysize != Lsize:
        print('accuracy()  Y,L 的维度不一致')
        return False,False
    # 标签转换为 int 类型
    L = L.int()

    correct = 0
    number = Ysize
    true_false_array = np.zeros(Ysize,dtype=np.int)
    for i,(y,l) in enumerate(zip(Y,L)):
        # 以0.5为阈值归为0或者1
        y = y.ge(0.5).int()
        if False not in torch.eq(y,l):
            correct += 1
            true_false_array[i] = 1
    acc = correct/number

    # true_false_array中,每一位置1表示该索引位置预测正确,置0表示预测错误
    return acc,true_false_array

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
        # 正向计算获得输出
        Y = model(X)
        # 与模型连接
        loss = lossFunc(Y, L)
        # 梯度初始化归零,准备优化
        optimizer.zero_grad()
        # 反向传播,更新梯度
        loss.backward()
        # 根据计算得到的梯度,结合优化器参数进行模型参数更新
        optimizer.step()
        # 获取损失值
        train_loss = loss.item()
        # 准确度
        train_acc,tfarr = accuracy(Y, L)
        if train_acc is not False:
            print('训练集\n    偏差为:{}\n    准确度:{}%'.format(train_loss,train_acc*100))

        # 模型转换为评估模式
        model.eval()
        # 正向计算获得输出
        Yt = model(Xt)
        # 计算损失
        tloss = lossFunc(Yt, Lt)
        # 获取损失值
        test_loss = tloss.item()
        # 准确度
        test_acc,tfarr = accuracy(Yt, Lt)
        if test_acc is not False:
            print('测试集\n    偏差为:{}\n    准确度:{}%'.format(test_loss, test_acc*100))
        print()


path = './Demo.pt'
def save():
    torch.save(model.state_dict(), path)        # 只保存模型参数,但不保存模型本身,因此在加载模型的时候要定义原模型类

def load(Xt,Lt):
    model = Model()                             # 实例化模型
    model.load_state_dict(torch.load(path))     # 加载模型参数
    model.eval()
    Yt = model(Xt)
    acc = accuracy(Yt,Lt)
    print(acc)


if __name__ == '__main__':

    miu1 = np.array([[-5,0]])
    miu2 = np.array([[5,0]])

    n = 100
    Xn1 = np.random.randn(n, 2) + miu1
    Ln1 = np.zeros((n, 2))
    Ln1[:, 0] = 1.0
    Xn2 = np.random.randn(n, 2) + miu2
    Ln2 = np.zeros((n, 2))
    Ln2[:, 1] = 1.0

    Xn = np.r_[Xn1,Xn2]
    Ln = np.r_[Ln1,Ln2]

    Xtrain = torch.from_numpy(Xn).float()        # 训练集样本
    Ltrain = torch.from_numpy(Ln).float()        # 训练集标签

    n = 25
    Xn1 = np.random.randn(n, 2) + miu1
    Ln1 = np.zeros((n, 2))
    Ln1[:, 0] = 1.0
    Xn2 = np.random.randn(n, 2) + miu2
    Ln2 = np.zeros((n, 2))
    Ln2[:, 1] = 1.0

    Xn = np.r_[Xn1, Xn2]
    Ln = np.r_[Ln1, Ln2]

    Xtest = torch.from_numpy(Xn).float()       # 测试集样本
    Ltest = torch.from_numpy(Ln).float()       # 测试集标签

    p.scatter(Xtrain[:, 0], Xtrain[:, 1], color='blue')
    p.scatter(Xtest[:, 0], Xtest[:, 1], color='red')
    p.xlim(-10,10)
    p.ylim(-10,10)
    p.grid()
    # p.show()

    train(Xtrain, Ltrain, Xtest, Ltest)         # 训练
    # save()                                    # 保存
    # load(Xtest, Ltest)                        # 加载




















