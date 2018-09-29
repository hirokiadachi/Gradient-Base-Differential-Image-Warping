import os
import csv
import copy
import cupy as xp
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

import chainer
import chainer.functions as F
from chainer import cuda
from chainer import Variable
from MATLAB_code.getcamK import getcamK
from MATLAB_code.computeRT import computeRT
from MATLAB_code.compute3Dpositions import compute3Dpositions
from transform import projective_inverse_warp
from extracte_euler_angle import Rotation2Euler, Rot2Fixed_xyz, Rot2Fixed_zyx
from make_movie_from_images import make_movie

def choose_target_image(dirname, tgt_num=200):
    img_name = 'scene_00_{:0>4}.png'.format(tgt_num)
    path = os.path.join(dirname, img_name)
    img = Image.open(path)
    img_mat_np = np.asarray(img, dtype='f')
    img_mat_np = img_mat_np.transpose(2, 0, 1)
    img_mat = xp.asarray(img, dtype='f')
    img_mat = img_mat.transpose(2, 0, 1)

    return img_mat, img_mat_np

def make_sequence_src_img(dirpath, rand_num=3, seq_num=3):
    img_name = 'scene_00_{:0>4}.png'.format(rand_num)
    path = os.path.join(dirpath, img_name)
    img = Image.open(path)
    img_mat = xp.asarray(img)
    img_mat = xp.asarray(img_mat, dtype='f').transpose(2,0,1)
    img_mat = xp.expand_dims(img_mat, axis=0)
    sequence_img_matrix = xp.expand_dims(img_mat, axis=0)

    return sequence_img_matrix

def save_images(filename, item, transposed=False):
    img = np.transpose(item, (1,2,0))
    img = img.astype(np.int32)
    img = np.uint8(img)
    img = Image.fromarray(img)
    img.save(filename)

def synthesizing_image(filename, base_item, paste_item):
    base_item = np.squeeze(base_item, axis=0)
    # transforming Base image
    img_base = np.transpose(base_item, (1,2,0))
    img_base = np.uint8(img_base)
    img_base = Image.fromarray(img_base)
    # transforming Paste image
    img_paste = np.transpose(paste_item, (1,2,0))
    img_paste = np.uint8(img_paste)
    img_paste = Image.fromarray(img_paste)
    img_paste.putalpha(160)
    img_paste.convert('RGBA')
    img_base.paste(img_paste, (0, 0), img_paste)
    img_base.save(filename)

def main(tgt_num=0, src_num=200, save_dir='./results'):
    """
    target image poses (predicted poses)
    predicted poses --> R
    predicted depth --> stacked_tgt_depth
    """
    Max_iter = 20000
    #base_number = 100
    #t_plus1 = base_number + 20
    #t_minus1 = base_number - 20
    dir_path = './ICL-NUIM'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    depth = compute3Dpositions(dir_path, tgt_num, xp=xp).astype('f')
    depth = xp.expand_dims(depth, axis=0)
    tgt_imgs, tgt_imgs_np = choose_target_image(dir_path, tgt_num)
    src_imgs = make_sequence_src_img(dir_path,
                                     rand_num=src_num,
                                     seq_num=3)

    K = getcamK(dir_path, tgt_num, xp=xp)
    curr_intrinsics = Variable(xp.expand_dims(K.astype('f'), axis=0))

    B, S, _, H, W = src_imgs.shape
    stacked_src_imgs = xp.reshape(src_imgs, (B, -1, H, W))
    stacked_tgt_depth = xp.reshape(depth, (B, -1, H, W))

    tgt_depth = F.reshape(stacked_tgt_depth, (B, 1, -1))
    tgt_depth = F.broadcast_to(tgt_depth, (B, 3, tgt_depth.shape[2]))
    #tgt_imgs = np.reshape(imgs_rgb[0, 0], (-1, H, W))
    save_images(os.path.join(save_dir,'tgt_img{}.png').format(tgt_num), tgt_imgs_np)
    curr_src_imgs = F.resize_images(stacked_src_imgs, (H, W))
    source_img = np.squeeze(curr_src_imgs.data, axis=0)
    source_img = chainer.cuda.to_cpu(source_img)
    filename = os.path.join(save_dir, 'source_image.png')
    save_images(filename, source_img)

    ## 6DoF value
    expand = xp.asarray([[0, 0, 0, 1]], dtype=np.float32)
    R_T_world2tgtcam  = xp.vstack([chainer.cuda.to_gpu(computeRT(dir_path, tgt_num)),
                                                        expand])
    R_T_tgtcam2world = xp.linalg.inv(R_T_world2tgtcam)
    R_T_world2srccam = xp.vstack([chainer.cuda.to_gpu(computeRT(dir_path, src_num)),
                                                       expand])
    R_T_srccam2world = xp.linalg.inv(R_T_world2srccam)

    ## target ---> source
    ## M = M_source * M_target'
    GT_RT = []
    R_t_cam1_to_tgtcam = xp.matmul(R_T_world2tgtcam, R_T_srccam2world)
    #R_t_cam1_to_tgtcam = xp.matmul(R_T_world2tgtcam,
    #                               R_T_srccam2world)
    R = R_t_cam1_to_tgtcam[:3, :3]
    t = R_t_cam1_to_tgtcam[:3, 3:].transpose()
    R = Rotation2Euler(R, 'zyx')[0]
    R = xp.expand_dims(R, axis=0).astype('f')
    R = chainer.cuda.to_gpu(R)
    R_t = xp.concatenate([R, t], axis=1).astype('f')
    R_t = Variable(R_t)
    GT_RT.append(R_t.data)
    with open(os.path.join(save_dir, 'gt_RT.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(R_t.data)

    perturbation = [0.1, 0.1, 0.001, 0.1, 0.1, 0.2]
    for ind, val in enumerate(perturbation):
        print(val)
        R_t.data[0][ind] += val
    R_t_noise = Variable(R_t.data)
    tgt_imgs  = Variable(xp.expand_dims(tgt_imgs, axis=0))
    src_imgs  = F.reshape(src_imgs, (B, -1, H, W))


    ## Differenciable image warp
    ERROR = {}
    lr = 1e-5 ## Learning Rate
    error_copy = 0
    for i in range(Max_iter):
        ## arguments : target image, depth of target image,
        ## (R|t) + a little noise, K (intrinsics parameter)
        curr_proj_img_ = projective_inverse_warp(src_imgs,
                                                 tgt_depth, R_t_noise,
                                                 curr_intrinsics)
        ## calicurating gradient of Error
        if i >= 5000:
            lr = 5e-6
        elif i >= 10000:
            lr = 1e-6

        curr_proj_error = F.mean_absolute_error(tgt_imgs,
                                                curr_proj_img_)

        difference = curr_proj_error - error_copy
        #if difference.data > 1 and i > 0:
        #if i > 10000:
        #    lr = 1e-5
        #else:lr = 1e-4
        #elif difference.data == 0 and i > 0:
        #    lr = .5e-4
        if curr_proj_error.data < 0.5 and i > 0:break

        error_copy = copy.deepcopy(curr_proj_error)
        SHAPE = curr_proj_error.shape
        curr_proj_error.grad = xp.ones(SHAPE, 'f')
        curr_proj_error.backward()
        R_t_grad = R_t_noise.grad
        R_t_noise = Variable(R_t_noise.data - (lr * R_t_grad))
        ERROR[i] = curr_proj_error.data

        # Plot the Mean Abusolute Error
        if i % 1000 == 0:

            plt.figure()
            plt.xlabel('Iteration')
            plt.ylabel('Error')
            plt.plot(ERROR.values(), label='Mean Abusolute Error')
            plt.legend()
            plt.savefig('results_graph/tough.jpg')

            print('{} iteration\nmin value: {}'\
                  .format(i, curr_proj_error.data))
            ## Saving result images
            curr_proj_img = np.squeeze(curr_proj_img_.data, axis=0)
            trans_save_dir_name = 'transposed_images'
            trans_save_dir_path = os.path.join(save_dir,
                                               trans_save_dir_name)
            syns_save_dir_name = 'synthesizing_images'
            syns_save_dir_path = os.path.join(save_dir,
                                              syns_save_dir_name)
            if not os.path.exists(trans_save_dir_path):
                os.makedirs(trans_save_dir_path)
            if not os.path.exists(syns_save_dir_path):
                os.makedirs(syns_save_dir_path)

            filename = os.path.join(trans_save_dir_path,
                                   'transposed_image{:0>6}.png'.format(i))
            save_images(filename, chainer.cuda.to_cpu(curr_proj_img), transposed=True)
            filename = os.path.join(syns_save_dir_path,
                                   'synthesizing_image{:0>6}.png'.format(i))
            synthesizing_image(filename, chainer.cuda.to_cpu(tgt_imgs.data),
                               chainer.cuda.to_cpu(curr_proj_img))

        elif i % 10 == 0:
            print('{} iteration\nmin value: {}'\
                  .format(i, curr_proj_error.data))
            ## Saving result images
            curr_proj_img = np.squeeze(curr_proj_img_.data, axis=0)
            trans_save_dir_name = 'transposed_images'
            trans_save_dir_path = os.path.join(save_dir,
                                               trans_save_dir_name)
            syns_save_dir_name = 'synthesizing_images'
            syns_save_dir_path = os.path.join(save_dir,
                                              syns_save_dir_name)
            if not os.path.exists(trans_save_dir_path):
                os.makedirs(trans_save_dir_path)
            if not os.path.exists(syns_save_dir_path):
                os.makedirs(syns_save_dir_path)

            filename = os.path.join(trans_save_dir_path,
                                   'transposed_image{:0>6}.png'.format(i))
            save_images(filename, chainer.cuda.to_cpu(curr_proj_img), transposed=True)
            filename = os.path.join(syns_save_dir_path,
                                   'synthesizing_image{:0>6}.png'.format(i))
            synthesizing_image(filename, chainer.cuda.to_cpu(tgt_imgs.data), 
                               chainer.cuda.to_cpu(curr_proj_img))

            ## Saving updated R and T parameter
            with open(os.path.join(save_dir, 'transpose_R_T.csv'), 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(R_t_noise.data)

if __name__ == '__main__':
    main()
    make_movie('results/synthesizing_images', 'GIF_syn', 'tough_syn.gif')
    make_movie('results/transposed_images', 'GIF_trans', 'tough_trans.gif')
