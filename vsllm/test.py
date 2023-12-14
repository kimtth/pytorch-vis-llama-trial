import os

current_file_path = os.path.realpath(__file__)

# get the directory of the current file
current_file_dir = os.path.dirname(current_file_path)


pkl_path = os.path.join(current_file_path, f'train.pkl')
print(pkl_path)


import torch
avl = torch.cuda.is_available()
print(avl)