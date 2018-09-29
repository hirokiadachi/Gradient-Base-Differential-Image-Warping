import os
import numpy as np
import matplotlib as m
m.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation as A
from PIL import Image


import matplotlib as mpl

def layout():
    c_cycle = ("#3498db", "#e74c3c", "#1abc9c", "#9b59b6", "#f1c40f", "#ecf0f1", "#34495e",
               "#446cb3", "#d24d57", "#27ae60", "#663399", "#f7ca18", "#bdc3c7", "#2c3e50")
    mpl.rc('font', size=10)
    mpl.rc('lines', linewidth=2, color="#2c3e50")
    mpl.rc('patch', linewidth=0, facecolor="none", edgecolor="none")
    mpl.rc('text', color='#2c3e50')
    # mpl.rc('axes', facecolor='none', edgecolor="none", titlesize=20, labelsize=15, color_cycle=c_cycle, grid=False)
    mpl.rc('axes', titlesize=15, labelsize=12)
    mpl.rc('xtick.major', size=10, width=0)
    mpl.rc('ytick.major', size=10, width=0)
    mpl.rc('xtick.minor', size=10, width=0)
    mpl.rc('ytick.minor', size=10, width=0)
    mpl.rc('ytick', direction="out")
    mpl.rc('grid', color='#c0392b', alpha=0.5, linewidth=1)
    mpl.rc('legend', fontsize=25, markerscale=1, labelspacing=0.2, frameon=True, fancybox=True,
           handlelength=0.1, handleheight=0.5, scatterpoints=1, facecolor="#eeeeee")
    mpl.rc('figure', figsize=(10, 6), dpi=224, facecolor="none", edgecolor="none")
    mpl.rc('savefig', dpi=1500)

def make_movie(img_file, save_dir_gif='./GIF', save_gif_filename=''):
    img_list = os.listdir(img_file)
    img_list = sorted(img_list)

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['left'].set_color('None')
    ax.spines['bottom'].set_color('None')
    ax.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    IMG = []
    if not os.path.exists(save_dir_gif):
        os.makedirs(save_dir_gif)

    for name in img_list:
        path = os.path.join(img_file, name)
        print(path)
        img = Image.open(path)
        IMG.append([plt.imshow(img, interpolation="spline36")])

    ani = A(fig, IMG, interval=40, repeat_delay=1000)
    savename = os.path.join(save_dir_gif, save_gif_filename)
    ani.save(savename, writer='imagemagick')
    print('===== FINISH ======')

if __name__ == '__main__':
    filename = './results/synthesizing_images'

    make_movie(filename)
