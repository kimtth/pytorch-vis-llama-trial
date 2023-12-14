"""
    This file provides a case for multi-modal dataset construction. 
"""
import os 
import torch 
from transformers import CLIPProcessor, CLIPModel 
from PIL import Image 
import pickle
import argparse  
import json 
from tqdm import tqdm 


def coco_process(args, clip_model, processor): 

    data_year_tag = '2017'
    dev_tag = ''
    if args.dev == True:
        dev_tag = 'dev'

    with open(os.path.join(args.dataset_path, f'annotations/captions_train{data_year_tag}.json'), 'r') as f: 
        data = json.load(f)['annotations']
    
    all_embeddings = [] 
    all_texts = []
    len_data = len(data)
    for i in tqdm(range(len_data)): 
        d = data[i] 
        img_id = d['image_id']
        img_id_fill = str(img_id).zfill(12)
        filename = os.path.join(args.dataset_path, f"train{data_year_tag}{dev_tag}/{img_id_fill}.jpg")

        if args.dev == True and os.path.exists(filename) == False: 
            continue

        image = Image.open(filename)
        inputs = processor(images=image, return_tensors='pt') 

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs) 
        all_embeddings.append(image_features) 
        all_texts.append(d['caption']) 
        if args.dev == True and i > 100:
            print(i, filename)
            break
    
    # Get the path of current file
    current_file_path = os.path.realpath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    pkl_path = os.path.join(current_file_dir, f'train.pkl')
    with open(pkl_path, 'wb') as f: 
        pickle.dump({'image_embedding': torch.cat(all_embeddings, dim=0), 
                        'text': all_texts}, f)


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--clip_path', default='./clip')
    parser.add_argument('--dataset_path', default='../COCO')
    parser.add_argument('--dataset_type', default='COCO') 
    parser.add_argument('--dev', default=True)

    # Manually set the arguments
    clip_path = os.path.join(os.getcwd(), 'model', 'clip-vit-large-patch14')
    dataset_path = os.path.join(os.getcwd(), 'dataset', 'coco2017')
    args = parser.parse_args(['--clip_path', clip_path, '--dataset_path', dataset_path])

    # args = parser.parse_args()
    clip_model = CLIPModel.from_pretrained(args.clip_path)
    processor = CLIPProcessor.from_pretrained(args.clip_path) 

    if args.dataset_type == 'COCO': 
        coco_process(args, clip_model, processor) 
    else:
        pass 


if __name__ == '__main__': 
    main()

