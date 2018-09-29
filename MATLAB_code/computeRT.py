import os
import cupy as xp
import numpy as np
import math

def computeRT(item_path, xp=np):
    with open(item_path, 'r') as f:
        LINES = [line.strip("';\n") for line in f]

    param = {}
    for index, args in enumerate(LINES):
        name = args[:15].replace(" ", "")
        name = name[:].replace("=", "")
        val  = args[15:]
        val = xp.asarray(val.strip("[]").split(","), dtype=xp.float64)
        param[name] = val

    cam_dir = param["cam_dir"]
    cam_pos = param["cam_pos"]
    cam_up  = param["cam_up"]

    z = cam_dir / xp.linalg.norm(cam_dir)

    x = xp.cross(cam_up, z)
    x = x / xp.linalg.norm(x)

    y = xp.cross(z, x)

    x = xp.expand_dims(x, axis=1)
    y = xp.expand_dims(y, axis=1)
    z = xp.expand_dims(z, axis=1)

    R = xp.concatenate([x, y, z], axis=1)
    T = xp.expand_dims(cam_pos, axis=1)

    R_T = xp.concatenate([R, T], axis=1)

    return R_T

if __name__ == '__main__':
    dirpath = '/data/Desktop/ICL-NUIM'
    file_number = 0
    R_T = computeRT(dirpath, file_number)
    print(R_T)
