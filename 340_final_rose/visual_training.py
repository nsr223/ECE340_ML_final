
#Noah Rose ECE 340 Final Project
import numpy as np
import time
import tkinter as tk
from skimage.draw import random_shapes
import matplotlib as plt

#parameters. Feel free to modify

EPISODE = 0
episode_history = []

done = False

width = 24
height = 24


width_cell = 20
height_cell = 20
margin_cell = 5
pos_x = 0
pos_y = 0

actions = ["up", "down", "left", "right"]

q_table = np.zeros((width, height, 4))

#colors for the tkinter
BLACK = "#000000"
WHITE = "#FFFFFF"
RED = "#FF0000"
GREEN = "#00FF00"
GREY = "#DCDCDC"

GRID_SIZE = width


REWARD_GOAL = 10
REWARD_WALL = -2
REWARD_BONUS = 2

total_reward = 0
avg_reward_history = []

learning_rate = 0.5

discount_factor = 0.5

epsilon = 0.1

steps = 0

#Initalize tkinter
window = tk.Tk()
window.title("Q-learning maze game")

canvas_width = (width_cell + margin_cell) * width + margin_cell
canvas_height = (height_cell + margin_cell) * height + margin_cell

canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
canvas.pack()



#creates a random maze to solve!
def random_shape_maze(width, height, max_shapes, max_size, allow_overlap, shape=None):
    x, _ = random_shapes([height, width], max_shapes, max_size=max_size, multichannel=False, shape=shape, allow_overlap=allow_overlap)
    
    x[x == 255] = 0
    x[np.nonzero(x)] = 1
    
    # wall
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1

    x = np.array(x)
    
    return x

def choose_action(state):
    if np.random.uniform() < epsilon:
        #chooses a random action if less than epsilon
        action = np.random.choice(actions)
    else:
        #chooses the action with the highest q-value
        action = actions[np.argmax(q_table[state])]
    return action

def generate_rewards(current_maze):
    reward_table = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            if current_maze[x][y] == 1:
                reward_table[x][y] = REWARD_WALL
            else:
                distance = np.sqrt((x - width - 2) ** 2 + (y - height - 2) ** 2)
                #sets the reward to be the goal reward minus the distance
                reward_table[x][y] = REWARD_GOAL - distance
    return reward_table



def update_q_table(state, next_state, action, reward):
    #calculates the maximum q-value for the next state
    max_q_next_state = np.max(q_table[next_state])
    #updates the q-value for the current state and action
    q_table[state][actions.index(action)] += learning_rate * (reward + discount_factor * max_q_next_state - q_table[state][actions.index(action)])


current_maze = random_shape_maze(width = width, height = height, max_shapes = 20, max_size = 4, allow_overlap = False, shape = None)
REWARD_STEP = generate_rewards(current_maze)

start_state = (1,1)
goal_state = (height -2, width -2)
state = start_state

while not done:

    EPISODE += 1   



    # draw the grid
    for row in range(len(current_maze)):
        for column in range(len(current_maze[0])):
            x1 = (margin_cell + width_cell) * column + margin_cell
            y1 = (margin_cell + height_cell) * row + margin_cell
            x2 = x1 + width_cell
            y2 = y1 + height_cell
            if current_maze[column][row] == 1:
                color = "black"
            else:
                color = "white"
            canvas.create_rectangle(x1, y1, x2, y2, fill=color)
            if row == len(current_maze) - 2 and column == len(current_maze) - 2:
                canvas.create_rectangle(x1, y1, x2, y2, fill="green")

    # draw the player
    x1 = (margin_cell + width_cell) * pos_x + margin_cell
    y1 = (margin_cell + height_cell) * pos_y + margin_cell
    x2 = x1 + width_cell
    y2 = y1 + height_cell
    canvas.create_rectangle(x1, y1, x2, y2, fill="red")

    # update the window
    window.update()
    time.sleep(0.5)
    

    action = choose_action(state)
    
    if action == "up" and state[0] > 0 and current_maze[state[0]-1][state[1]] == 0:
        next_state = (state[0]-1, state[1])
        reward = REWARD_STEP[state[0] - 1][state[1]]
    elif action == "down" and state[0] < GRID_SIZE-1 and current_maze[state[0]+1][state[1]] == 0:
        next_state = (state[0]+1, state[1])
        reward = REWARD_STEP[state[0] + 1][state[1]]
    elif action == "left" and state[1] > 0 and current_maze[state[0]][state[1]-1] == 0:
        next_state = (state[0], state[1]-1)
        reward = REWARD_STEP[state[0]][state[1]-1]
    elif action == "right" and state[1] < GRID_SIZE-1 and current_maze[state[0]][state[1]+1] == 0:
        next_state = (state[0], state[1]+1)
        reward = REWARD_STEP[state[0]][state[1]+1]
    else:
        next_state = state
        reward = REWARD_STEP[state[0]][state[1]]
    


    pos_x = next_state[0]
    pos_y = next_state[1]

    update_q_table(state, next_state, action, reward)

    state = next_state

    #setting up parameters for training plot:
    episode_history.append(EPISODE)
    total_reward = total_reward + reward
    avg_reward = total_reward / EPISODE
    avg_reward_history.append(avg_reward)






    if(state == goal_state):
        done = True
        print("Win!!")


#creating the training plots:
lr = str(learning_rate)
df = str(discount_factor)
tn = str(TRIAL_NUMBER)
CurrentDate=str(datetime.date.today())
plt.plot(episode_history, avg_reward_history)
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.title('Maze Game Training Plot Trial ' + tn +' Learning rate: ' + lr + ' Discount factor: '+ df)
plt.savefig(CurrentDate+ "_training" + tn +  "_lr_"+ lr + "_df_" + df + ".jpg")






