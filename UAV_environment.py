import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from numpy import random

class UAVenv(gym.Env):
    metadata = {'render.modes': ['human']}
    
  #User and UAV parameters
    NUM_USER = 100   
    NUM_UAV = 5 
    MAX_USER_COVER_EACH_UAV= 20  
    UAV_HEIGHT = 350
    BW_UAV = 4e6  # Total Bandwidth per UAV   # Update to decrease the available BW
    BW_RB = 180e3  # Bandwidth of a Resource Block
    ACTUAL_BW_UAV = BW_UAV * 0.9
    THETA = 60 * math.pi / 180
    #Grid specifications
    COVERAGE_XY = 1000
    grid_space = 100
    GRID_SIZE = int(COVERAGE_XY / grid_space)
    
    
    def pol2cart(r,theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return (x, y)
    
    def cart2pol(x, y):
        theta = np.arctan2(y, x)
        r = np.hypot(x, y)
        return r, theta
    
    
    #HOTSPOTS = np.array([[300, 300], [800, 800], [300, 800], [800, 300]])  
    #USER_DIS = int(NUM_USER / NUM_UAV)
    #USER_LOC = np.zeros((NUM_USER - USER_DIS, 2))
    
    
    #for i in range(len(HOTSPOTS)):
        #for j in range(USER_DIS):
            #temp_loc_r = np.random.uniform(-(1/5)*COVERAGE_XY, (1/5)*COVERAGE_XY)
            #temp_loc_theta = np.random.uniform(0, 2*math.pi)
            #temp_loc = pol2cart(temp_loc_r, temp_loc_theta)
            #(temp_loc_1, temp_loc_2) = temp_loc
            #temp_loc_1 = temp_loc_1 + HOTSPOTS[i, 0]
            #temp_loc_2 = temp_loc_2 + HOTSPOTS[i, 1]
            #USER_LOC[i * USER_DIS + j, :] = [temp_loc_1, temp_loc_2]
            
    #temp_loc = np.random.uniform(low=0, high=COVERAGE_XY, size=(USER_DIS, 2))
    #USER_LOC = np.concatenate((USER_LOC, temp_loc))
    #x = [i[0] for i in USER_LOC]
    #y = [i[1] for i in USER_LOC]
    #plt.scatter(x,y)
    #plt.show()
    #np.savetxt('UserLocation.txt', USER_LOC, fmt='%.3e', delimiter=' ', newline='\n')
    USER_LOC = np.loadtxt('UserLocation.txt', delimiter=' ').astype(np.int64)
    
    
    def __init__(self):
        super(UAVenv, self).__init__()
        # Defining action spaces // UAV RB allocation to each user increase each by 1 untill remains
        # Five different action for the movement of each UAV
        # 0 = Right, 1 = Left, 2 = straight, 3 = back, 4 = Hover
        # Defining Observation spaces // UAV RB to each user
        # Position of the UAV in space // X and Y pos
        self.u_loc = self.USER_LOC
        self.state = np.zeros((self.NUM_UAV, 2), dtype=np.int32)
        #print(self.state)
        # Set the states to the hotspots and one at the centre for faster convergence
        # Further complexity by choosing random value of state or starting at same initial position
        # self.state[:, 0:2] = [[1, 2], [4, 2], [7, 3], [3, 8], [4, 5]]
        # Starting UAV Position at the center of the target area
        #self.state[:, 0:2] = [[5, 5], [5, 5],[5, 5], [5, 5],[5, 5]]
        self.state[:, 0:2] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.flag = [0, 0, 0, 0, 0]
        self.coverage_radius = self.UAV_HEIGHT * np.tan(self.THETA / 2)
        #print(self.coverage_radius)

    def step(self, action):
        # Take the action
        # Assignment of sub carrier band to users
        # Reshape of actions
        # Execution of one step within the environment
        # Deal with out of boundaries conditions
        isDone = False
        # Calculate the distance of every users to the UAV BS and organize as a list
        
        dist_u_uav = np.zeros(shape=(self.NUM_UAV, self.NUM_USER))
        
        for i in range(self.NUM_UAV):
            temp_x = self.state[i, 0]
            temp_y = self.state[i, 1]
            #print("Before action: ", (temp_x, temp_y))
            # one step action
            #print(action)
            if action[i] == 0:
                self.state[i, 0] = self.state[i, 0] + 1
            elif action[i] == 1:
                self.state[i, 0] = self.state[i, 0] - 1
            elif action[i] == 2:
                self.state[i, 1] = self.state[i, 1] + 1
            elif action[i] == 3:
                self.state[i, 1] = self.state[i, 1] - 1
            elif action[i] == 4:
                pass
            else:
                print("Error Action Value")

            # Take boundary condition into account // Individual flag for punishing the UAV
            if self.state[i,0] < 0 or self.state[i,0] > self.GRID_SIZE or self.state[i, 1] < 0 or self.state[i,1] > self.GRID_SIZE:
                self.state[i, 0] = temp_x
                self.state[i, 1] = temp_y
                self.flag[i] = 1 
            else:
              self.flag[i] = 0             

            #print("After action: ", (self.state[i, 0], self.state[i, 1]))
            

            # Calculation of the distance value for all UAV and User
            for l in range(self.NUM_USER):
                dist_u_uav[i, l] = math.sqrt((self.u_loc[l, 0] - (self.state[i, 0] * self.grid_space)) ** 2 + (self.u_loc[l, 1] -(self.state[i, 1] * self.grid_space)) ** 2)
       
            max_user_num = self.ACTUAL_BW_UAV/ self.BW_RB
        ################ 
         # Initialize the connection_request table with zeros
        connection_request = np.zeros(shape=(self.NUM_UAV, self.NUM_USER), dtype="int")

        # For each user, find the closest UAV
        for i in range(self.NUM_USER):
            close_uav = np.argmin(dist_u_uav[:,i]) # Closest UAV index
            # If the distance between the UAV and user is within the coverage radius, then send the connection request
            if dist_u_uav[close_uav, i] <= self.coverage_radius:
                connection_request[close_uav, i] = 1 # Send the connection request

        # User association flag to keep track of which users are associated with which UAVs
        user_asso_flag = np.zeros(shape=(self.NUM_UAV, self.NUM_USER), dtype="int")
        # Total number of users associated with each UAV
        uav_asso_count = np.zeros(shape=(self.NUM_UAV,1), dtype="int")
        #for i in range(self.NUM_UAV):
            #print("UAV number:", i, "has a count of", uav_asso_count[i][0])

        # Allocate users to UAVs
        for i in range(self.NUM_UAV):
        # Find the users who have sent the connection request to the current UAV
            temp_user = np.where(connection_request[i, :] == 1)
            # Sort the users based on their distance from the UAV
            temp_user_distance = dist_u_uav[i, temp_user]
            temp_user_sorted = np.argsort(temp_user_distance)
            # Convert temp_user to np_array so that it can be indexed easily
            temp_user = np.array(temp_user)
            # Get the actual index of the users who have sent the connection request, sorted based on the distance from the UAV
            temp_user_actual_idx = temp_user[0, temp_user_sorted]
            # Set the user association flag for each UAV and closest user
            for user_index in temp_user_actual_idx[0]:
                # If the number of users associated with the UAV is less than the maximum number of users it can cover, then allocate the user
                if uav_asso_count[i] < self.MAX_USER_COVER_EACH_UAV:
                    user_asso_flag[i, user_index] = 1
                    uav_asso_count[i] += 1
                else:
                    break
        
        # For the second sweep, sweep through all users
        # If the user is not associated choose the closest UAV and check whether it has any available resource
        # If so allocate the resource and set the User association flag bit of that user to 1
        for j in range(self.NUM_USER):
            if not(np.any(user_asso_flag[:, j] != 0)):
                close_uav_id = dist_u_uav[:, j]
                close_uav_id = [i[0] for i in sorted(enumerate(close_uav_id), key=lambda x: x[1])]
                if dist_u_uav[close_uav_id[0], j] <= self.coverage_radius:
                    for close_id in close_uav_id:
                        if np.sum(user_asso_flag[close_id,:]) < max_user_num:
                            user_asso_flag[close_id, j] = 1
                            break

        # Calculation of reward function  
        sum_user_assoc = np.sum(user_asso_flag, axis = 1)
        #print(sum_user_assoc)
        reward_solo = np.zeros(np.size(sum_user_assoc), dtype="float32")
        for k in range(self.NUM_UAV):
            if self.flag[k] != 0:
                reward_solo[k] = np.copy(sum_user_assoc[k] - 2)
                isDone = True
            else:
                reward_solo[k] = np.copy(sum_user_assoc[k]) 
        reward = np.copy(reward_solo)
        #print(reward)

                                                                                                               
        # Return of obs, reward, done, info
        return np.copy(self.state).reshape(1, self.NUM_UAV * 2), reward, isDone, "empty", sum(sum_user_assoc)
    
    def render(self, ax, mode='human', close=False):
        # Implement viz
        if mode == 'human':
            ax.cla()
            position = self.state[:, 0:2] * self.grid_space
            print(self.state)
            ax.scatter(self.u_loc[:, 0], self.u_loc[:, 1], c = '#00008b', marker='o',s=20, label = "Users")
            ax.scatter(position[:, 0], position[:, 1], c = '#000000', marker='s', label = "UAV")
            # ax.scatter(self.hotspot_loc[:, 0], self.hotspot_loc[:, 1], marker="*", s=100, c='red',label = "Hotspots")  
            for (i,j) in (position[:,:]):
                cc = plt.Circle((i,j), self.coverage_radius, alpha=0.1,facecolor='red')
                ax.set_aspect(1)
                ax.add_artist(cc)
            ax.legend()
            plt.pause(0.5)
            plt.xlim(-50, 1050)
            plt.ylim(-50, 1050)
            plt.draw()

    def reset(self):
        # reset out states
        # set the states to the hotspots and one at the centre for faster convergence
        # further complexity by choosing random value of state
        # self.state[:, 0:2] = [[1, 2], [4, 2], [7, 3], [3, 8], [4, 5]]
        self.state[:, 0:2] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        # Starting UAV Position at the center of the target area
        #self.state[:, 0:2] = [[5, 5], [5, 5],[5, 5], [5, 5],[5, 5]]
        return self.state

    def get_state(self):
        state_loc = np.zeros((self.NUM_UAV, 2))
        for k in range(self.NUM_UAV):
            state_loc[k, 0] = self.state[k, 0]
            state_loc[k, 1] = self.state[k, 1]
        return state_loc