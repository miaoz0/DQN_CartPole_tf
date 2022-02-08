import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self, model, gamma=None, lr=None) -> None:
        """ DQN algorithm
        Args:
            model (parl.Model)  - 定义Q函数的前向网络结构
            gamma (float)       - reward的衰减因子
            lr (float)          - learning_rate，学习率.
        """
        # checks
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model

        self.target_model = tf.keras.models.clone_model(model)
        # 将target_model设置为 不可训练
        self.target_model.trainable = False

        self.gamma = gamma
        self.lr = lr

        self.loss_func = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def predict(self, obs):
        """使用self.model来获取[Q(s,a1),Q(s,a2),...]
        """
        return self.model(obs)
    
    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        参数：
            batch_size 个 obs, action, reward, next_obs, terminal
        """
        with tf.GradientTape() as tape:
            # 获取Q预测值
            pred_values = self.model(obs)
            # 得到action的one-hot向量
            act_dim = pred_values.shape[-1]
            action = np.squeeze(action)             # (n,1) -> (n,)
            action_onehot = tf.one_hot(indices=action, depth=act_dim)

            # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
            # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
            #  ==> pred_value = [[3.9]]
            pred_values = pred_values * action_onehot
            pred_values = tf.reduce_sum(pred_values,axis=-1,keepdims=True)

            # 从target_model中 获得 max Q' 的值，用于计算 target_Q
            max_Q = self.target_model(next_obs)
            # max_Q: (n, act_dim) -> (n,1)
            max_Q = tf.reduce_max(max_Q,axis=-1,keepdims=True)        
            target_value = reward + (1 - terminal) * self.gamma * max_Q

            # 计算 Q(s,a) 与 target_Q的均方差，得到loss
            loss = self.loss_func(target_value, pred_values)
        # 计算梯度
        grads = tape.gradient(loss, self.model.trainable_variables)
        # 更新参数
        self.optimizer.apply_gradients(grads_and_vars=zip(grads,self.model.trainable_variables))

        return loss

    def sync_target(self):
        """把 self.model的参数 同步到 self.target_model
        """
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)
