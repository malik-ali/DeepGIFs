from builtins import range
from past.builtins import xrange
from math import sqrt, ceil
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import time
import imageio

def plot_frame_double(index, real, fake):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    fig.subplots_adjust(hspace=0)
    
    real = real.numpy()
    real = np.transpose(real, (1, 2, 0))
    
    fake = fake.numpy()
    fake = np.transpose(fake, (1, 2, 0))
    
    axes[0].imshow(real)
    axes[0].set_title(f'[REAL] Frame #{index + 1}')
    
    axes[1].imshow(fake)
    axes[1].set_title(f'[FAKE] Frame #{index + 1}')
    plt.gcf().show() 
    
def display_loop(gif, generated):
    for index, (frame1, frame2) in enumerate(zip(gif, generated)):
        plot_frame_double(index, frame1, frame2)

        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.15) 

def plot_frame_single(index, frame):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0)
    npframe = frame.numpy()
    npframe = np.transpose(npframe, (1, 2, 0))
    ax.imshow(npframe)
    ax.set_title(f'FRAME #{index + 1}')
    plt.gcf().show()

def img_show(img, save_file=None):
    npimg = img.numpy()
    out = np.transpose(npimg, (1, 2, 0))
    plt.figure(figsize=(20,40))
    if save_file:
        plt.imsave(save_file, out)
    plt.imshow(out)
    
def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G

def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H,W,C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N*H+N, D*W+D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming)/(maxg-ming)
    return G

def save_as_gif(out_path, tensor):
    tensor = np.array(tensor)
    tensor = tensor.squeeze()
#     tensor = tensor.astype(np.uint8)
    imageio.mimsave(out_path, tensor, 'gif')
