# -*-coding:utf-8-*-
"""
カメラ内部変数を導出するソースコード
matlabのソースコードgetcamK.mを参考に作成
"""

import os
import cupy as xp
import numpy as np
import math

def getcamK(item_path, xp=np):
    with open(item_path, 'r') as f:
        LINES = [line.strip("';\n") for line in f]

    param = {}
    for index, args in enumerate(LINES):
        name = args[:15].replace(" ", "")
        name = name[:].replace("=", "")
        val  = args[15:]
        val = xp.asarray(val.strip("[]").split(","),
                                    dtype=xp.float64)
        param[name] = val

    focal  = xp.linalg.norm(param["cam_dir"])
    aspect = xp.linalg.norm((param["cam_right"]) /
                             xp.linalg.norm(param["cam_up"]))
    angle  = 2*math.atan((xp.linalg.norm(param["cam_right"]) / 2) /
                                  xp.linalg.norm(param["cam_dir"]) )

    height = M = 480 ## cam_height
    width  = N = 640 ## cam_width

    # pixel size
    psx = 2 * focal * math.tan(0.5*angle)/N
    psy = 2 * focal * math.tan(0.5*angle)/aspect/M

    Sx = psx = psx / focal
    Sy = psy = psy /focal

    Ox = (width+1)*0.5
    Oy = (height+1)*0.5

    f = focal

    K = xp.asarray([[1/psx, 0, Ox],
                    [0, 1/psy, Oy],
                    [0,   0,    1]], dtype=xp.float64)
    K[1,1] = -K[1,1]

    return K

if __name__ == '__main__':
    dirpath = '/data/Desktop/ICL-NUIM'
    file_number = 0
    K = getcameraK(dirpath, file_number)
