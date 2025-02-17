import sys
import os
sys.path.insert(0, os.path.abspath('./open_clip_local'))
import open_clip_local as open_clip
from datasets import load_dataset
import torch
from tqdm import tqdm
import webdataset as wds
import os
import pandas as pd
from multiprocessing import Pool
from functools import partial
import tarfile
import io
import logging
from PIL import Image
from torch.utils.data import Dataset
import sys
sys.path.append('/root/open_clip_local/src/')
from open_clip_train.data import get_data
import open_clip_train

class TarCsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.image_tar_paths = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.tokenize = tokenizer
        
        # Initialize tar_files at init time
        self.tar_files = {}
        
    def get_image_from_tar(self, tar_path):
        # Parse tar path and image name from combined string
        tar_file, img_name = tar_path.split('@')
        
        # Cache tar file handles
        if tar_file not in self.tar_files:
            self.tar_files[tar_file] = tarfile.open(tar_file, 'r')
        
        tar = self.tar_files[tar_file]
        img_member = tar.getmember(img_name)
        img_file = tar.extractfile(img_member)
        img_bytes = io.BytesIO(img_file.read())
        return Image.open(img_bytes)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = self.get_image_from_tar(str(self.image_tar_paths[idx]))
        image = self.transforms(image)
        text = self.tokenize([str(self.captions[idx])])[0]
        return image, text
        
    def __del__(self):
        if hasattr(self, 'tar_files'):
            for tar in self.tar_files.values():
                tar.close()

def process_tar_file(tar_path):
    data = []
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()
            members.sort(key=lambda x: x.name)
            
            for i in range(0, len(members)-1, 2):
                if i+1 >= len(members):
                    break
                    
                m1, m2 = members[i], members[i+1]
                if m1.name.endswith('.jpg') and m2.name.endswith('.txt'):
                    txt_content = tar.extractfile(m2).read().decode('utf-8').strip()
                    data.append({
                        'filepath': f'{tar_path}@{m1.name}',
                        'caption': txt_content
                    })
    except Exception as e:
        print(f"Error processing {tar_path}: {e}")
        return []
    
    return data

def create_webdataset(dataset, output_dir, split):
    """Convert HuggingFace dataset to WebDataset format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create shards of 1000 samples each
    samples_per_shard = 1000
    split_data = dataset[split]
    
    for shard_idx in tqdm(range(0, len(split_data), samples_per_shard), desc=f"Creating {split} webdataset"):
        with wds.TarWriter(f"{output_dir}/{split}_shard_{shard_idx:06d}.tar") as dst:
            end_idx = min(shard_idx + samples_per_shard, len(split_data))
            for idx in range(shard_idx, end_idx):
                item = split_data[idx]
                image = item['image']
                text = item['text']
                
                # Save as webdataset sample
                dst.write({
                    "__key__": f"sample_{idx}",
                    "jpg": image,
                    "txt": text.encode('utf-8')
                })

def create_csv_mapping(data_dir, split):
    tar_files = [os.path.join(data_dir, f) 
                 for f in sorted(os.listdir(data_dir)) 
                 if f.endswith('.tar')]
    
    # Use number of CPU cores for parallelization
    with Pool() as pool:
        all_data = pool.map(process_tar_file, tar_files)
    
    # Flatten the list of lists
    flat_data = [item for sublist in all_data for item in sublist]
    
    # Create DataFrame and save
    df = pd.DataFrame(flat_data)
    output_file = f'{split}_mapping.csv'
    df.to_csv(output_file, index=False)
    print(f"Processed {len(flat_data)} {split} image-text pairs")
    return output_file

def main(create_data: bool = False):
    if create_data:
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset("sae-rad/MIMIC_chexpert_padchest_clip")
        print("Dataset loaded")
        print(f"Train set size: {len(dataset['train'])}")
        print(f"Test set size: {len(dataset['test'])}")
        
        # Create output directories
        train_dir = 'chest_xray_webdataset_train'
        test_dir = 'chest_xray_webdataset_test'
        
        # Create train and test webdatasets
        print("Creating train webdataset...")
        create_webdataset(dataset, train_dir, 'train')
        
        print("Creating test webdataset...")
        create_webdataset(dataset, test_dir, 'test')
        
        # Create CSV mappings
        print("Creating train CSV mapping...")
        train_csv = create_csv_mapping(train_dir, 'train')
        
        print("Creating test CSV mapping...")
        test_csv = create_csv_mapping(test_dir, 'test')
    else:
        train_csv = '/root/train_mapping.csv'
        test_csv = '/root/test_mapping.csv'
    
    # Get dataset sizes
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Training arguments
    args = [
        '--model', 'coca_ViT-L-14',
        '--pretrained', 'laion2B-s13B-b90k',
        '--train-data', train_csv,
        '--val-data', test_csv,
        '--train-num-samples', str(len(train_df)),
        '--val-num-samples', str(len(test_df)),
        '--dataset-type', 'csv',
        '--csv-separator', ',',
        '--csv-img-key', 'filepath',
        '--csv-caption-key', 'caption',
        '--batch-size', '200',
        '--lr', '1e-5',
        '--wd', '0.1',
        '--epochs', '50',
        '--workers', '4',
        '--coca-contrastive-loss-weight', '1',
        '--coca-caption-loss-weight', '1',
        '--precision', 'amp',
        '--report-to', 'wandb',
        '--wandb-project-name', 'coca-chest-xray',
        '--name', 'coca-finetune',
        '--wandb-notes', 'Finetuning CoCa on chest x-ray images',
        '--log-every-n-steps', '100',
        '--save-frequency', '5',
        '--val-frequency', '1',
        '--zeroshot-frequency', '0',
        '--logs', './logs'
    ]

    # Call training main
    from open_clip_train.main import main as train_main
    train_main(args)

if __name__ == "__main__":
    open_clip_train.data.CsvDataset = TarCsvDataset
    main()