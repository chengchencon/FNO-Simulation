# import all modules and our model layers
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys,os 
import matplotlib.pyplot as plt
from utilities3 import *
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
#from Adam import Adam
from tensorflow.keras.optimizers import Adam
torch.manual_seed(3407)
np.random.seed(0)
#parallel training 20230501
from torch import distributed as dist
torch.set_num_threads(1)
