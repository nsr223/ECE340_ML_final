ECE 340 FINAL PROJECT
NOAH ROSE

MAKE SURE TO HAVE THE FOLLOWING IMPORTED/INSTALLED:


import numpy as np
import time
import tkinter as tk
from skimage.draw import random_shapes
import matplotlib.pyplot as plt
import datetime


(These are all common except for skimage.draw. To install skimage use

python -m pip install -U scikit-image)


Maze solving with q-learning.  

If you want to change the learning rate, discount factor, or epsilon you must go and edit it in the code.

You can also edit other parameters, such as the maze width and height, if desired.

This project contains 3 .py files:

1. visual_training.py shows you one training session, fully animated with tkinter. Note that if your computer doesn't have a good graphics card, this may freeze up a lot.

2. quick_train.py will run one training session and create a training plot. This doesn't show any animation and is much, much quicker.

3. train_and_results.py is the most complete file. You can train the agent multiple times (though 3-5 is recommended) and generate training plots for each round of training and after that you have the option to watch the trained agent solve the maze. This is also animated with tkinter, but because the agent is trained there shouldn't be any rendering issues.


Note: The mazes generated are random, and sometimes don't generate many obstacles. If you want more obstacles, just abort and run the code again.