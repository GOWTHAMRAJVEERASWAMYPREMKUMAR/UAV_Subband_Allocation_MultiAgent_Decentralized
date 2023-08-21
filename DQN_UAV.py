from ast import Num
import random
import numpy as np
import math
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from UAV_environment import UAVenv
from misc import final_render
from collections import deque
import torch
from torch import Tensor, nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import os
from scipy.io import savemat

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Define a neural network class for reinforcement learning

class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()
        
        # Save state and action sizes as instance variables
        self.state_size = state_size
        self.action_size = action_size
        
        # Define a sequential stack of linear and ReLU layers
        self.linear_stack = nn.Sequential(
            nn.Linear(self.state_size, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, self.action_size)
        )
        
        # Move the model to the appropriate device (GPU or CPU)
        self.linear_stack.to(device)
        
    def forward(self, x):
        # Move the input tensor to the appropriate device
        x = x.to(device)
        
        # Pass the input tensor through the network's layers
        Q_values = self.linear_stack(x)
        
        # Return the Q-values
        return Q_values

# Define a Deep Q-Network class for reinforcement learning

class DQL:
    def __init__(self, state_size=2, action_size=5, discount_factor=0.95, epsilon=.10, alpha=2.5e-4):
        # Initialize instance variables
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=125000)
        self.gamma = discount_factor
        self.epsilon = epsilon
        #self.epsilon_decay = epsilon_decay
        self.learning_rate = alpha
        
        # Create the main and target networks using the NeuralNetwork class
        self.main_network = NeuralNetwork(self.state_size, self.action_size).to(device)
        self.target_network = NeuralNetwork(self.state_size, self.action_size).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        # Define the optimizer and loss function for training the main network
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        self.loss_func = nn.SmoothL1Loss()  # Huber Loss
        
        # Set the number of steps taken so far to 0
        self.steps_done = 0

    def store_transition(self, state, action, reward, next_state, done):
        # Store the transition in the replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state):
    # Determine the exploration rate (epsilon) using an epsilon decay policy
    #epsilon = self.epsilon * math.exp(-1 * self.steps_done / self.epsilon_decay)
        temp=random.random()
        #self.steps_done += 1

    # Choose a random action with probability epsilon, or the action with the highest Q-value with probability (1-epsilon)
        if  temp < self.epsilon:
            action = torch.tensor([[np.random.randint(0, self.action_size)]], device=device, dtype=torch.long)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.main_network(state_tensor)
                action = q_values.argmax().unsqueeze(0)

        return action

    def train(self, batch_size):
    # Sample minibatch from replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

       # Convert minibatch data to tensors and move to GPU if available
        state = torch.FloatTensor(np.vstack([x[0] for x in minibatch])).to(device)
        action = torch.LongTensor(np.vstack([x[1] for x in minibatch])).to(device)
        reward = torch.FloatTensor(np.vstack([x[2] for x in minibatch])).to(device)
        next_state = torch.FloatTensor(np.vstack([x[3] for x in minibatch])).to(device)
        done = torch.Tensor(np.vstack([x[4] for x in minibatch])).to(device)

        with torch.no_grad():
          next_Q_values = self.target_network(next_state)
          next_Q_max = next_Q_values.max(1)[0]
          reward = reward.view(next_Q_max.size())
          done = done.view(next_Q_max.size())
          target_Q = reward + self.gamma * next_Q_max * (1 - done)
          

       # Compute Q-value estimates for the current state and the selected action
        Q_main = self.main_network(state).gather(1, action).squeeze()

      # Compute the loss function between the estimated Q-value and the target Q-value
        loss_func = nn.SmoothL1Loss()
        loss = loss_func( Q_main, target_Q.float())

      # Zero-out the gradients from the previous iteration and backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()

      # Clip the gradients to avoid exploding gradients
        nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1)

      # Update the network weights using the computed gradients
        self.optimizer.step()


  
u_env = UAVenv()
GRID_SIZE = u_env.GRID_SIZE
NUM_UAV = u_env.NUM_UAV
NUM_USER = u_env.NUM_USER
num_episode = 1000
num_epochs = 100
#discount_factor = 0.95
#alpha = 3.5e-4
batch_size = 512
update_rate = 10  #50
#dnn_epoch = 1
epsilon = 0.10
#epsilon_min = 0.10
#epsilon_decay = 1
random.seed(SEED)




# Keeping track of the episode reward
episode_reward = np.zeros(num_episode)
episode_user_connected = np.zeros(num_episode)

fig = plt.figure()
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])

UAV_OB = [None, None, None, None, None]


for k in range(NUM_UAV):
            UAV_OB[k] = DQL()
best_result = 0

for i_episode in range(num_episode):
    print(i_episode)

    # Environment reset and get the states
    u_env.reset()

    # Get the initial states
    states = u_env.get_state()
    reward = np.zeros(NUM_UAV)

    
    for t in range(num_epochs):
        
        # Determining the actions for all drones
        # update the target network
        drone_act_list = []
        states_ten = torch.from_numpy(states)
        for k in range(NUM_UAV):
            state = states_ten[k, :]
            action = UAV_OB[k].epsilon_greedy(state.float())
            drone_act_list.append(action)
            if t % update_rate == 0:
                UAV_OB[k].target_network.load_state_dict(UAV_OB[k].main_network.state_dict())

        # Find the global reward for the combined set of actions for the UAV
        temp_data = u_env.step(drone_act_list)
        reward = temp_data[1]
        done = temp_data[2]
        next_state = u_env.get_state()

        # Store the transition information
        for k in range(NUM_UAV):
            ## Storing of the information on the individual UAV and it's reward value in itself.
            state = states_ten[k, :]
            action = drone_act_list[k]
            next_sta = next_state[k, :]
            reward_ind = reward[k]
            UAV_OB[k].store_transition(state, action, reward_ind, next_sta, done)

        # Calculation of the total episodic reward of all the UAVs 
        # Calculation of the total number of connected User for the combination of all UAVs
        episode_reward[i_episode] += np.sum(reward)
        episode_user_connected[i_episode] += temp_data[4]

        states = next_state

        for k in range(NUM_UAV):
            if len(UAV_OB[k].replay_buffer) > batch_size:
                UAV_OB[k].train(batch_size)
    
    if i_episode % 10 == 0:
        # Reset of the environment
        u_env.reset()
        # Get the states
        states = u_env.get_state()
        states_ten = torch.from_numpy(states)
        for t in range(100):
            drone_act_list = []
            for k in range(NUM_UAV):
                state = states[k,:]
                state = torch.unsqueeze(torch.FloatTensor(state),0)
                Q_values = UAV_OB[k].main_network.forward(state.float())
                best_next_action = torch.max(Q_values, 1)[1].data.numpy()
                best_next_action = best_next_action[0]
                drone_act_list.append(best_next_action)
            temp_data = u_env.step(drone_act_list)
            states = u_env.get_state()
            states_fin = states
            # print(temp_data)
            if best_result < temp_data[4]:
                best_result = temp_data[4]
                best_state = states
        u_env.render(ax1)
        plt.title("Intermediate state of UAV in this episode")
        print(drone_act_list)
        print("Number of user connected in ",i_episode," episode is: ", temp_data[4])

def smooth(y, pts):
    box = np.ones(pts)/pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

## Save the data from the run as a file
mdict = {'num_episode':range(0, num_episode),'episodic_reward': episode_reward}
savemat('Results\\episodic_reward.mat', mdict)

# Plot the accumulated reward vs episodes
fig = plt.figure()
plt.plot(range(0, num_episode), episode_reward)
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Episode vs Episodic Reward")
plt.show()
fig = plt.figure()
plt.plot(range(0, num_episode), episode_user_connected)
plt.xlabel("Episode")
plt.ylabel("Connected User in Episode")
plt.title("Episode vs Connected User in Epsisode")
plt.show()
fig = plt.figure()
smoothed = smooth(episode_reward, 10)
plt.plot(range(0, num_episode-10), smoothed[0:len(smoothed)-10] )
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Smoothed Epidode vs Episodic Reward")
plt.show()
fig = plt.figure()
final_render(states_fin, "final")
fig = plt.figure()
final_render(best_state, "best")
print(states_fin)
print('Total Connected User in Final Stage', temp_data[4])
print("Best State")
print(best_state)
print("Total Connected User (Best Outcome)", best_result)