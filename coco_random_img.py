import os
import random
import shutil

def select_random_files(directory, num_files):
    all_files = os.listdir(directory)
    random_files = random.sample(all_files, num_files)
    return random_files

dataset_parent_directory = os.path.join(os.getcwd(), 'dataset')
src_directory = os.path.join(dataset_parent_directory, 'coco2017', 'train2017') 
destination_directory = os.path.join(dataset_parent_directory, 'coco2017', 'train2017dev') 
num_files = 1000
random_files = select_random_files(src_directory, num_files)

for file in random_files:
    filename = os.path.join(src_directory, file)          
    shutil.copy(filename, destination_directory)