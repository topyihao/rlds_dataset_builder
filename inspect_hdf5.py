#!/usr/bin/env python3
import h5py
import numpy as np

# Open one of the HDF5 files to inspect its structure
with h5py.File('/home/allied/aloha_data/aloha_preprocessed/put_sponge_into_pot/train/episode_0.hdf5', 'r') as f:
    print('Keys in HDF5 file:')
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f'  {name}: shape={obj.shape}, dtype={obj.dtype}')
        else:
            print(f'  {name}/')
    f.visititems(print_structure)
    
    # Check if the image shapes match the existing dataset builder expectations
    if '/observations/images/cam_high' in f:
        print(f'\nImage shape: {f["/observations/images/cam_high"].shape}')
    
    # Check action and state dimensions
    if '/action' in f:
        print(f'Action shape: {f["/action"].shape}')
    if '/observations/qpos' in f:
        print(f'State (qpos) shape: {f["/observations/qpos"].shape}') 