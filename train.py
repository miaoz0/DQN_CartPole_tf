import gym
import logging
logging.basicConfig(level=logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import sys
sys.path.append(r"code\Reforcement-Learning\DQN-CartPole")

from agent import Agent
from model import MyModel
from algorithm import DQN
from replay_memory import ReplayMemory


LEARN_FREQ = 5              # 训练频率，不需要每一个step都learn，积攒一些Experience后再learn，提高效率
MEMORY_SIZE = 200000        # replay memory大小，越大约占内存
MEMORY_WARMUP_SIZE = 200    # replay memory里需要先 预存一些Experience（再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 64             # batch size
LEARNING_RATE = 0.0005      # 学习率
GAMMA = 0.99                # reward衰减因子，一般取0.9~0.999不等

def run_train_episode(agent,env,rpm):
    """
    参数：
        agent           - 智能体
        env             - 环境
        rpm             - replay memory
    返回：
        total_reward    - 总奖励
    """
    total_reward = 0
    obs = env.reset()       # 初始化环境
    step = 0
    while True:
        step += 1
        obs = np.expand_dims(obs,axis=0)        # 拓展一个维度
        action = agent.sample(obs)              # 尝试一个动作
        obs = np.squeeze(obs)
        next_obs, reward, done, info = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        # print("#step: ",step)
        # print("obs: ",obs)
        # print("act: ",action)
        # print("reward: ",reward)
        # print("next_obs: ",next_obs)
        # print("terminal: ",done)
        # print("len(rpm): ",len(rpm))
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s, a, r, s', done
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward

def run_evaluate_episodes(agent, env, render=False):
    """评估agent，跑n个episode，求平均reward

    参数：
        agent           - 智能体
        env             - 环境
        render          - 是否渲染出环境
    返回：
        平均 【评估奖励】
    """
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        
        while True:
            obs = np.expand_dims(obs,axis=0)        # 拓展一个维度
            action = agent.predict(obs)             # 预测动作，只选择最优动作
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def main(env_name="MountainCar-v0"):
    # CartPole-v0: expected reward > 180
    # MountainCar-v0 : expected reward > -120
    env = gym.make(env_name)
    
    obs_dim = env.observation_space.shape[0]        # CartPole-v0: (4,)
    act_dim = env.action_space.n                    # CartPole-v0: 2

    rpm = ReplayMemory(MEMORY_SIZE)

    model = MyModel(obs_dim=obs_dim,act_dim=act_dim)
    algorithm = DQN(model, gamma=GAMMA, lr= LEARNING_RATE)
    agent = Agent(
        algorithm,
        act_dim,
        e_greed=0.1,
        e_greed_decrement=1e-6
    )

    # 加载模型
    save_file_name = f"code/Reforcement-Learning/DQN-CartPole/weights/{env_name}/dqn"
    # agent.restore(save_file_name)

    # 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)

    max_episode = 2000

    # 开始训练
    episode = 0
    while episode < max_episode:
        # train
        for i in range(50):
            total_reward = run_train_episode(agent, env, rpm)
            episode += 1
        
        # test
        eval_reward = run_evaluate_episodes(agent, env, render=False)
        logger.info(f'episode:{episode}    e_greed:{agent.e_greed}   Test reward:{eval_reward}')

    # 训练结束，保存模型
    # agent.save(save_file_name)

def evalutate(env_name="MountainCar-v0"):
    """使用保存的权重 测试
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]        # CartPole-v0: (4,)
    act_dim = env.action_space.n                    # CartPole-v0: 2

    rpm = ReplayMemory(MEMORY_SIZE)

    model = MyModel(obs_dim=obs_dim,act_dim=act_dim)
    algorithm = DQN(model, gamma=GAMMA, lr= LEARNING_RATE)
    agent = Agent(
        algorithm,
        act_dim,
        e_greed=0.1,
        e_greed_decrement=1e-6
    )

    # 加载模型
    save_file_name = f"code/Reforcement-Learning/DQN-CartPole/weights/{env_name}/dqn"
    agent.restore(save_file_name)

    eval_reward = run_evaluate_episodes(agent, env, render=True)
    logger.info(f'Test reward:{eval_reward}')

if __name__ == "__main__":
    main(env_name="CartPole-v0")
    # evalutate(env_name="CartPole-v0")