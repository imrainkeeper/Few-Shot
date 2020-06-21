import h5py
import torch
import shutil
import os
import numpy as np


def save_checkpoint(state, checkpoint_save_dir, epoch, train_accuracy, val_accuracy):
    checkpoint_filepath = os.path.join(checkpoint_save_dir, 'checkpoint' + '_' + str(epoch) + '_' + str(train_accuracy) + '_' + str(val_accuracy) + '.pth.tar')
    torch.save(state, checkpoint_filepath)
