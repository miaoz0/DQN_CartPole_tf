import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.activations import relu
import numpy as np

class MyModel(tf.keras.Model):
    """使用全连接网络
    
    参数：
        obs_dim     - observation dimension
        act_dim     - action dimension
    """
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # 全连接层的神经单元数
        hidden_size = 128

        self.input_layer = Input(shape=obs_dim)  # 输入
        # 3层全连接层
        self.fc1 = Dense(units=hidden_size,activation=relu)
        self.fc2 = Dense(units=hidden_size,activation=relu)
        self.fc3 = Dense(units=act_dim)                     # 无激活函数

        self.out = self.call(self.input_layer)              # 输出

        # reinitialize
        super().__init__(
            inputs = self.input_layer,
            outputs = self.out,
            name = "my_model"
        )
    
    # 前向传播
    def call(self, inputs, training=False):
        X = self.fc1(inputs)
        X = self.fc2(X)
        Q = self.fc3(X)
        return Q

if __name__ == "__main__":
    model = MyModel(1,5)
    # model.summary()