import numpy as np
from model import MyModel

class Agent():
    def __init__(self, algorithm, act_dim, e_greed=0.1, e_greed_decrement=0) -> None:
        """
        参数：
            algorithm           - object    算法
            act_dim             - int       动作总数
            e_greed             - float     随机探索概率
            e_greed_decrement   - float     随机探索概率 每步衰减值
        """
        assert isinstance(act_dim, int)     # 断言：act_dim一定为int类型
        
        self.alg = algorithm

        self.act_dim = act_dim

        self.global_step = 0
        self.update_target_steps = 200      # 每200个step更新target模型

        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement
    
    def sample(self, obs):
        """根据观测值obs，采样（带探索）一个动作
        参数：
            obs         - 观测值
        返回：
            act         - 动作 [0,act_dim)
        """
        p = np.random.random()              # 0~1之间的 小数

        if p < self.e_greed:
            # 随机选择一个动作
            act = np.random.randint(self.act_dim)
        else:
            act = self.predict(obs)         # 选择最优的动作
        
        # 随着训练的进行，随机选择的概率 逐渐下降
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)

        return act
    
    def predict(self, obs):
        """根据观测值obs，选择最优的动作
        """
        pred_q = self.alg.predict(obs)
        act = np.argmax(pred_q)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        """ 根据训练参数 更新一次模型参数
        """
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)
        # print("obs: ",obs.shape)                  # (n, 4)
        # print("act: ",act.shape)                  # (n, 1)
        # print("reward: ",reward.shape)            # (n, 1)
        # print("next_obs: ",next_obs.shape)        # (n, 4)
        # print("terminal: ",terminal.shape)        # (n, 1)
        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return loss.numpy()

    def save(self, file_name):
        """保存模型
        """
        self.alg.model.save_weights(file_name+"-model.h5",save_format="h5")
        self.alg.target_model.save_weights(file_name+"-target_model.h5",save_format="h5")

    def restore(self, file_name):
        """加载模型
        """
        self.alg.model.load_weights(file_name+"-model.h5")
        self.alg.target_model.load_weights(file_name+"-target_model.h5")