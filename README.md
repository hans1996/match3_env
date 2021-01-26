# match3_env


## Getting started
### Installing
```bash
git clone https://github.com/hans1996/match3_env.git
cd gym-match3
pip install -e .
```

### example


```python
from gym_match3.envs import Match3Env
import gym

env = Match3Env() 
for i_episode in range(1): #玩一次遊戲
    observation = env.reset()
    for t in range(10):  #做10個 action
        print(observation)
        action = env.action_space.sample()
        print('action: ',action)
        observation, reward, done, info = env.step(action)
        
        print('reward:', reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

```

```python
env.get_available_actions()   #查看action對應的動作
```

For more information on `gym` interface visit [gym documentation](https://gym.openai.com/docs/)

```
