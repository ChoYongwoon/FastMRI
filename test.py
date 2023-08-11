from utils.data.load_data import create_data_loaders
import argparse
import shutil
from utils.learning.train_part import train
from pathlib import Path
import cv2

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix

def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=20, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_Unet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='C:\FastMRI/Data/train/image/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='C:\FastMRI/Data/val/image/', help='Directory of validation data')
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--input-key', type=str, nargs='+', default=['image_input','image_grappa'], help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args
    
def main():
    recon_path = '../../Data/train/reconstructions_train/brain_acc4_1.h5'
    #recon_data = glob.glob(os.path.join(recon_path,'*.h5'))
    
    with h5py.File(recon_path, 'r') as h5_file:
        print(h5_file.keys())
    
    
    
    
    '''
    leaderboard_data_path1 = 'C:\FastMRI/Data/leaderboard/acc4/image/'
    leaderboard_data1 = glob.glob(os.path.join(leaderboard_data_path1,'*.h5'))
    
    
    leaderboard_data_path2 = 'C:\FastMRI/Data/leaderboard/acc8/image/'
    leaderboard_data2 = glob.glob(os.path.join(leaderboard_data_path2,'*.h5'))
    
    
    input_data_path = 'C:\FastMRI/Data/train/image/'
    val_data_path = 'C:\FastMRI/Data/val/image/'
    input_data = glob.glob(os.path.join(input_data_path,'*.h5'))
    val_data = glob.glob(os.path.join(val_data_path,'*.h5'))
    
    leader_input_means = []
    leader_grappa_means = []
    leader_target_means = []
    for filename in leaderboard_data1:
        with h5py.File(filename, 'r') as h5_file:
            image = np.array(h5_file['image_input'])
            image1 = np.array(h5_file['image_grappa'])
            image2 = np.array(h5_file['image_label'])
            leader_input_means.append(np.mean(image))
            leader_grappa_means.append(np.mean(image1))
            leader_target_means.append(np.mean(image2))
            
    for filename in leaderboard_data2:
        with h5py.File(filename, 'r') as h5_file:
            image = np.array(h5_file['image_input'])
            image1 = np.array(h5_file['image_grappa'])
            image2 = np.array(h5_file['image_label'])
            leader_input_means.append(np.mean(image))
            leader_grappa_means.append(np.mean(image1))
            leader_target_means.append(np.mean(image2))
            
    train_input_means = []
    train_grappa_means = []
    train_target_means = []
    for filename in input_data:
        with h5py.File(filename, 'r') as h5_file:
            image = np.array(h5_file['image_input'])
            image1 = np.array(h5_file['image_grappa'])
            image2 = np.array(h5_file['image_label'])
            train_input_means.append(np.mean(image))
            train_grappa_means.append(np.mean(image1))
            train_target_means.append(np.mean(image2))
            
    val_input_means = []
    val_grappa_means = []
    val_target_means = []
    for filename in val_data:
        with h5py.File(filename, 'r') as h5_file:
            image = np.array(h5_file['image_input'])
            image1 = np.array(h5_file['image_grappa'])
            image2 = np.array(h5_file['image_label'])
            val_input_means.append(np.mean(image))
            val_grappa_means.append(np.mean(image1))
            val_target_means.append(np.mean(image2))
    
            
    mean_factor = 1.4
    aug_train_input_means = []
    aug_train_grappa_means = []
    aug_train_target_means = []
    for filename in input_data:
        with h5py.File(filename, 'r') as h5_file:
            image = np.array(h5_file['image_input'])
            image1 = np.array(h5_file['image_grappa'])
            image2 = np.array(h5_file['image_label'])
            scale = np.random.rand()+0.9
            aug_train_input_means.append(np.mean(image*scale))
            aug_train_grappa_means.append(np.mean(image1*scale))
            aug_train_target_means.append(np.mean(image2*scale))
        
    aug_val_input_means = []
    aug_val_grappa_means = []
    aug_val_target_means = []
    for filename in input_data:
        with h5py.File(filename, 'r') as h5_file:
            image = np.array(h5_file['image_input'])
            image1 = np.array(h5_file['image_grappa'])
            image2 = np.array(h5_file['image_label'])
            scale = np.random.rand()+0.9
            aug_val_input_means.append(np.mean(image*scale))
            aug_val_grappa_means.append(np.mean(image1*scale))
            aug_val_target_means.append(np.mean(image2*scale))
    
    
    print(leader_input_means)
    print(leader_grappa_means)
    print(leader_target_means)
    
    print(train_input_means)
    print(train_grappa_means)
    print(train_target_means)
    
    print(val_input_means)
    print(val_grappa_means)
    print(val_target_means)
    
    print(aug_train_input_means)
    print(aug_train_grappa_means)
    print(aug_train_target_means)
    
    print(aug_val_input_means)
    print(aug_val_grappa_means)
    print(aug_val_target_means)
    
    print(np.mean(leader_input_means)/np.mean(train_input_means))
    print(np.mean(leader_grappa_means)/np.mean(train_grappa_means))
    print(np.mean(leader_target_means)/np.mean(train_target_means))
    
    print(np.mean(leader_input_means)/np.mean(aug_train_input_means))
    print(np.mean(leader_grappa_means)/np.mean(aug_train_grappa_means))
    print(np.mean(leader_target_means)/np.mean(aug_train_target_means))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Draw the histograms
    axes[0,0].hist(leader_input_means, bins=20, alpha=0.5, label='Leaderboard_input')
    axes[0,0].hist(train_input_means, bins=20, alpha=0.5, label='Input_input')
    axes[0,0].hist(val_input_means, bins=20, alpha=0.5, label='Val_input')
    axes[0,0].set_xlabel('Mean')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('Histogram of Image Input Means')
    axes[0,0].legend(loc='upper right')
    
    axes[0,1].hist(leader_grappa_means, bins=20, alpha=0.5, label='Leaderboard_grappa')
    axes[0,1].hist(train_grappa_means, bins=20, alpha=0.5, label='Input_grappa')
    axes[0,1].hist(val_grappa_means, bins=20, alpha=0.5, label='Val_grappa')
    axes[0,1].set_xlabel('Mean')
    axes[0,1].set_title('Histogram of Image Grappa Means')
    axes[0,1].legend(loc='upper right')
    
    axes[0,2].hist(leader_target_means, bins=20, alpha=0.5, label='Leaderboard_target')
    axes[0,2].hist(train_target_means, bins=20, alpha=0.5, label='Input_target')
    axes[0,2].hist(val_target_means, bins=20, alpha=0.5, label='Val_target')
    axes[0,2].set_xlabel('Mean')
    axes[0,2].set_title('Histogram of Image Target Means')
    axes[0,2].legend(loc='upper right')
    
    axes[1,0].hist(leader_input_means, bins=20, alpha=0.5, label='Leaderboard_input')
    axes[1,0].hist(aug_train_input_means, bins=20, alpha=0.5, label='Input_input')
    axes[1,0].hist(aug_val_input_means, bins=20, alpha=0.5, label='Val_input')
    axes[1,0].set_xlabel('Mean')
    axes[1,0].set_ylabel('Count')
    axes[1,0].set_title('Histogram of Image Input Means')
    axes[1,0].legend(loc='upper right')
    
    axes[1,1].hist(leader_grappa_means, bins=20, alpha=0.5, label='Leaderboard_grappa')
    axes[1,1].hist(aug_train_grappa_means, bins=20, alpha=0.5, label='Input_grappa')
    axes[1,1].hist(aug_val_grappa_means, bins=20, alpha=0.5, label='Val_grappa')
    axes[1,1].set_xlabel('Mean')
    axes[1,1].set_title('Histogram of Image Grappa Means')
    axes[1,1].legend(loc='upper right')
    
    axes[1,2].hist(leader_target_means, bins=20, alpha=0.5, label='Leaderboard_target')
    axes[1,2].hist(aug_train_target_means, bins=20, alpha=0.5, label='Input_target')
    axes[1,2].hist(aug_val_target_means, bins=20, alpha=0.5, label='Val_target')
    axes[1,2].set_xlabel('Mean')
    axes[1,2].set_title('Histogram of Image Target Means')
    axes[1,2].legend(loc='upper right')
    
    plt.show()
'''

    
    
if __name__ == "__main__":
    main()