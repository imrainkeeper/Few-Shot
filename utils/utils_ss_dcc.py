import h5py
import torch
import shutil
import os
import numpy as np


def save_checkpoint(state, checkpoint_save_dir, epoch, train_accuracy_distance, train_accuracy_crowd_count, val_accuracy_distance, val_accuracy_crowd_count):
    checkpoint_filepath = os.path.join(checkpoint_save_dir, 'checkpoint' + '_' + str(epoch) + '_' + str(train_accuracy_distance) + '_' + str(train_accuracy_crowd_count) + '_' + str(val_accuracy_distance) + '_' + str(val_accuracy_crowd_count) + '.pth.tar')
    torch.save(state, checkpoint_filepath)
