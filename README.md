> 代码 参考修改自：
> [PARL实现DQN，CartPole环境](https://github.com/PaddlePaddle/PARL/tree/develop/examples/tutorials/parl2_dygraph/lesson3/dqn)


> 参考视频：
> [世界冠军带你从零实践强化学习](https://www.bilibili.com/video/BV1yv411i7xd?p=10&spm_id_from=pageDriver)



# DQN的两大创新点
1. 经验回放（Experience Repaly）
2. 固定Q目标（Fixed Q Target）

## 经验回放（Experience Repaly）

每个时间步agent与环境交互得到的转移样本存储在`buffer`中。

当进行模型参数的更新时，从`buffer`中随机抽取`batch_size`个数据，构造损失函数，利用梯度下降更新参数。


通过这种方式，
1. **去除数据之间的相关性**，缓和了数据分布的差异。
2. **提高了样本利用率**，进而提高了模型学习效率。

---

 **为什么要去除数据之间的相关性？** 

> 参考：

> [关于强化学习中经验回放（experience replay）的两个问题？](https://www.zhihu.com/question/278182581) 

> [为什么机器学习中, 要假设我们的数据是独立同分布的?](https://www.zhihu.com/question/41222495)

 **理解1：** 确保数据是**独立同分布**的。这样，我们搭建的模型是**简单、易用**的。 

 **理解2：** 
在一般环境中，**智能体得到奖励的情况往往很少**。比如在n个格子上，只有1个格子有奖励。智能体在不断地尝试中，大多数情况是没有奖励的。如果没有`Experience Repaly`，只是每步都进行更新，模型可能会找不到“正确的道路”，陷入**局部最优**或**不收敛**情况。

---

## 固定Q目标（Fixed Q Target）
> 参考：
> [DQL: Dueling Double DQN, Prioritized Experience Replay, and fixed Q-targets（三下）](https://blog.csdn.net/mike112223/article/details/90796992) 【前面一点的内容】

在DQN中，损失函数的定义是，使`Q`尽可能地逼近`Q_target`。

在实际情况中，`Q`在变化，作为 **“标签”** 的`Q_target`也在不断地变化。它使得我们的**算法更新不稳定**，即输出的`Q`在不断变化，训练的损失曲线轨迹是震荡的。

---

DQN引入了`target_net`。具体来说，使用`value_net`输出`Q`值，使用`target_net`输出`Q_target`值。

> `target_net`与`value_net`具有相同的网络结构，但不共享参数。
1. 在一段时间内，`target_net`保持不变，只训练`value_net`。这样，相当于固定住“标签”`Q_target`，然后使用预测值`Q`不断逼近。
2. 一段时间过后，将`value_net`的权重 复制到 `target_net`上，完成`target_net`参数的更新。

通过这种方式，**一定程度降低了当前`Q`值和`target_Q`值的相关性，提高了算法稳定性。**
