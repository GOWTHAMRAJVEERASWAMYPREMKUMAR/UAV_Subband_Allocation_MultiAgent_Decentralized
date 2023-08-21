
from matplotlib import gridspec
import matplotlib.pyplot as plt
from UAV_environment import UAVenv
import numpy as np 
from matplotlib.gridspec import GridSpec
import math
env=UAVenv()
def final_render(state, remark):
    USER_LOC = np.loadtxt('UserLocation.txt', dtype=np.int32, delimiter=' ')
    u_loc = USER_LOC
    HOTSPOTS = np.array([[300, 300], [800, 800], [300, 800], [800, 300]])
    hotspot_loc = HOTSPOTS 
    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0:1, 0:1])
    grid_space = 100
    UAV_HEIGHT = 350
    THETA = 60 * math.pi / 180
    coverage_radius = UAV_HEIGHT * np.tan(THETA / 2)


    ax.cla()
    position = state[:, 0:2] * grid_space
    ax.scatter(u_loc[:, 0], u_loc[:, 1], c = '#00008b', marker='o',s=20, label = "Users")
    ax.scatter(position[:, 0], position[:, 1], c = '#000000', marker='s', label = "UAV")
    ax.scatter(hotspot_loc[:, 0], hotspot_loc[:, 1], marker="*", s=100, c='red',label="Hotspots") 
    for (i,j) in (position[:,:]):
        cc = plt.Circle((i,j), coverage_radius, alpha=0.1,facecolor='red')
        ax.set_aspect(1)
        ax.add_artist(cc)
    ax.legend()
    if remark == "best":
        plt.title("Best state of UAV")
    elif remark == "final":
        plt.title("Final state of UAV")
    plt.pause(0.5)
    plt.xlim(-50, 1050)
    plt.ylim(-50, 1050)
    plt.show()