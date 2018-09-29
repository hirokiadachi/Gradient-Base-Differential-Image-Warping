import os
import cupy as xp
import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from MATLAB_code.getcamK import getcamK

def compute3Dpositions(item_path, file_path, xp=np):
    K = getcamK(file_path, xp=xp)

    fx = K[0, 0]
    fy = K[1, 1]
    u0 = K[0, 2]
    v0 = K[1, 2]

    u = xp.tile(xp.arange(1, 641), (480, 1))
    v = xp.expand_dims(xp.arange(1, 481), axis=1)
    v = xp.tile(v, (1, 640))

    u_u0_by_fx = (u - u0) / fx
    v_v0_by_fy = (v - v0) / fy

    with open(item_path, 'r') as f:
        lines = f.read()
    lines = lines.split()
    str2mat = xp.array(lines, dtype=xp.float64)
    z = xp.reshape(str2mat, (480, 640))

    z = z / xp.sqrt(u_u0_by_fx**2 + v_v0_by_fy**2 + 1)

    x = ((u-u0)/fx)*z
    y = ((v-v0)/fy)*z

    return z

if __name__ == '__main__':
    dir_path = '/data/Desktop/ICL-NUIM'
    file_number = 0
    compute3Dpositions(dir_path, file_number)
