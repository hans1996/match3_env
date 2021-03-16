from gym_match3.envs import Match3Env
import gym
import matplotlib.pyplot as plt
# import pickle
from matplotlib import style
#from gym_match3.envs.levels import Match3Levels, Level

from PIL import Image
import cv2
import numpy as np

import torch
from torch import nn
from torchvision import transforms as T

from pathlib import Path
from collections import deque
import random, datetime, os, copy

from gym_match3.envs.levels import LEVELS #  default levels
from gym_match3.envs.levels import Match3Levels, Level

import time, datetime




width_hight = 5

n_shapesss = 6



def Getlevels(WidthAndHeight,shapes):
    LEVELS = [Level(WidthAndHeight,WidthAndHeight,shapes, np.zeros((WidthAndHeight,WidthAndHeight)).tolist())]
    return LEVELS

 

env = Match3Env(levels=Match3Levels(Getlevels(width_hight,n_shapesss)))


class One_hot(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        
    def observation(self, observation):
        observation = observation.reshape(width_hight ,width_hight ).astype(int)
        grid_onehot = np.zeros(shape=(env.n_shapes, width_hight , width_hight ))
        table = {i:i for i in range(0,n_shapesss)} 

        for i in range(width_hight):
            for j in range(width_hight ):
                grid_element = observation[i][j]
                grid_onehot[table[grid_element]][i][j]=1
                
        return grid_onehot

env = One_hot(env)

env.reset()
next_state, reward, done, info = env.step(action=0)
#print(f"{next_state.shape},\n {reward},\n {done},\n {info}")


class Match3:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=1000000)
        self.batch_size = 500

        self.exploration_rate = 0.5
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 10000  # min. experiences before training
        self.learn_every = 1   # no. of experiences between updates to Q_online
        self.sync_every = 10000   # no. of experiences between Q_target & Q_online sync

        self.save_every = 500000   # no. of experiences between saving Match3 Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # Match3's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = Match3Net(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.
    Inputs:
    state: A single observation of the current state, dimension is (state_dim)
    Outputs:
    action_idx : An integer representing which action Match3 will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
            

        # EXPLOIT
        else:
            state = state.__array__()
            if self.use_cuda:
                
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)

            state = state.unsqueeze(0)
 
            action_values = self.net(state.float(), model="online")
            #print(action_values)
            action_idx = torch.argmax(action_values).item()
            #print(action_idx)
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        Inputs:
        state (np.array),
        next_state (np.array),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    def td_estimate(self, state, action):
        current_Q = self.net(state.float(), model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state.float(), model="online")
        #print(next_state_Q)
        #print(next_state_Q.shape)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state.float(), model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward +  (1 - done.float())* self.gamma * next_Q).float()    



    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
                        #print(self.curr_step)
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state.float(), action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state.float(), done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
    

    def save(self):
        save_path = (
            self.save_dir / f"Match3_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Match3Net saved to {save_path} at step {self.curr_step}")

    
class Match3Net(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """
  

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=15, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=15, out_channels=10, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=5, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(5*((((width_hight - 4 ) + 1) - 2) + 1 )*((((width_hight - 4 ) + 1) - 2) + 1 ), 512),   # first conv2D layer feature map: (8-4) + 1 = 5   , second conv2D feature map: (5-2)+1 = 4  
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        
          


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)


        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
            #f"Reward {rewardd} - "
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()


def plot_obs(observation=None):
    
    d = {0: (255, 255, 255),   #白色
     1: (0, 255, 255),      #黃色
     2: (0, 0, 255) ,      #紅色
     3: (230,224,176) ,   #灰藍色 
     4: (0,0,0), #黑色  
     5: (100,100,100),
        6:(150,150,150),
        7:(200,200,200),
        8:(200,150,250),
        9:(250,200,150),
        10:(150,200,250)}              
    
    
    background = np.zeros((6,6,3), dtype=np.uint8) #黑色的背景

    for color in range(env.n_shapes):
        result = np.where(observation[color] == 1)
        
        listOfCoordinates= list(zip(result[0], result[1]))
        
        for cord in listOfCoordinates:
            background[cord] = d[color]     
            
    img = Image.fromarray(background, 'RGB')
     
    
    #img = img.resize((500, 800)) 
    cv2.imshow("image", np.array(img))
    
    cv2.waitKey(200)






use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

match = Match3(state_dim=(n_shapesss , width_hight, width_hight), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 1000000
for e in range(episodes):


    state = env.reset()      # initialize for first step 

    # Play the game!
    while True:

        #if e % 100 == 0:
        #    plot_obs(next_state)

        # Run agent on the state
        action = match.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)
        
        # Remember
        match.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = match.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done :
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=match.exploration_rate, step=match.curr_step)

