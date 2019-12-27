# %%
# 导入框架，建立学习模型
import mxnet
from mxnet import init, nd
from mxnet.gluon import nn
from mxnet import gluon

# 模型直接参考了lenet，一共两个卷积层，两个池化层，三个全连接层，都用交叉熵函数激活
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # Dense会默认将(批量大小, 通道, 高, 宽)形状的输入转换成
        # (批量大小, 通道 * 高 * 宽)形状的输入
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))

# 模拟输入，高和宽均为28的单通道数据样本，并逐层进行前向计算来查看每个层的输出形状。
X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

# %%
# 导入数据集
batch_size = 256
train_iter, test_iter = mxnet.load_data_fashion_mnist(
    batch_size=batch_size)  # TODO: 改成官方数据集，或者自己摄像头识别后的数据集（需要预处理数据集）

ctx = mxnet.try_gpu()
# %%
# 训练模型
lr, num_epochs = 0.9, 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
mxnet.train_ch5(net, train_iter, test_iter,
                batch_size, trainer, ctx, num_epochs)
