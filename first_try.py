from __future__ import print_function

import numpy as np
import os
import time
import argparse

from agents import *
from utils import *
from results_proc import *

import gym
import gym_hanoi


if __name__ == "__main__":
    env = gym.make("Hanoi-v0")
    env.reset()
