import os
from datasets import load_dataset

full_size_dataset_tag = "HuggingFaceM4/COCO"
small_size_dataset_tag = "RIW/small-coco-wm_50_2"

dataset = load_dataset(full_size_dataset_tag)

download_path = os.path.join(os.getcwd(), "dataset")
dataset.save_to_disk(download_path)