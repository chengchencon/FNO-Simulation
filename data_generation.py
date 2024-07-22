#import necessary libraries
import torch
import matplotlib.pyplot as plt
from utilities3 import *
import numpy as np

# read necessary original wind simulation data
import matplotlib.pyplot as plt
import numpy as np
import os
TRAIN_PATH="data/Niigata_west_2m.npy" 
data = np.load(TRAIN_PATH)

# make sure the time index is at the last (400,400, 1200) 1200 is the time steps in total
data = np.transpose(data,(1,2,0))
#print(data.shape)

# here is the selected region of niigata data
data=data[72:328,72:328,180:]

# Generate the random coordinates
import random, pickle

def generate_all_coordinates(x_start, x_end, y_start, y_end):
    all_coords = []
    for x in range(x_start, x_end + 1):  # 包含x_end
        for y in range(y_start, y_end+1):  # 不包含y_end
            all_coords.append((x, y))
    return all_coords

all_coords = generate_all_coordinates(0,192,0,192) #192+64 = 256
# choose your wished location to be selected
location_select = 80
unique_coords = random.sample(all_coords, location_select)


# sampling the data into patches


subarrays = []
timestep = 1 



# decide the resolution of each patch
patch_length = 64
trunk_len = 30
final_time = 1020-trunk_len
step = 2
for coord in unique_coords:
    x = coord[0]
    y = coord[1]
    for k in range(0,final_time,step):
        subarray = data[x:x+patch_length, y:y+patch_length, k:k+trunk_len]
        subarrays.append(subarray)       
    
DataSliced = np.array(subarrays)



np.save("data/your_wishedName.npy", DataSliced)
