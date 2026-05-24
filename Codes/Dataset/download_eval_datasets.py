import os
import sys
import json
import logging
import pandas as pd
from datasets import load_dataset

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.env_loader import env_loader

logger = setup_logging('download_eval_datasets')

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
EVAL_DIR = os.path.join(DATA_DIR, 'datasets_eval')

def download_multi_crows_pairs():
    """Download Multi-lingual CrowS-Pairs (English subset)."""
    out_dir = os.path.join(EVAL_DIR, 'multicrows')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'crows_pair_english.csv')
    
    if os.path.exists(out_path):
        logger.info(f"Multi-CrowS-Pairs already exists at {out_path}")
        return
        
    logger.info("Downloading Multi-lingual CrowS-Pairs (English)...")
    dataset = load_dataset("HuggingFaceM4/Multi-lingual-crows-pairs", "english", split="test")
    df = dataset.to_pandas()
    
    # Standardize column names for downstream evaluation scripts
    if 'sent_more' in df.columns and 'sent_less' in df.columns:
        df.rename(columns={'sent_more': 'stereo', 'sent_less': 'anti'}, inplace=True)
        
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} pairs to {out_path}")

def download_indian_bias():
    """Download Indian-BhED (Indian Multilingual Bias English)."""
    out_dir = os.path.join(EVAL_DIR, 'indian_bias')
    os.makedirs(out_dir, exist_ok=True)
    
    logger.info("Downloading Indian-BhED dataset...")
    # This dataset has multiple files or splits typically.
    # We will try to load the main dataset. If it's structured as one dataset with a 'category' column,
    # we'll save it as one CSV or split it.
    try:
        dataset = load_dataset("Aksht/Indian-Multilingual-Bias-English", split="train")
        df = dataset.to_pandas()
        
        # Standardize columns
        if 'sent_more' in df.columns and 'sent_less' in df.columns:
            df.rename(columns={'sent_more': 'stereo', 'sent_less': 'anti'}, inplace=True)
            
        out_path = os.path.join(out_dir, 'indian_bias_english.csv')
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df)} pairs to {out_path}")
        
    except Exception as e:
        logger.error(f"Failed to download Indian-BhED from HF: {e}")
        # The exact repo structure might require different handling, but this is the requested repo name.

def download_winobias():
    """Download WinoBias dataset."""
    out_dir = os.path.join(EVAL_DIR, 'winobias')
    os.makedirs(out_dir, exist_ok=True)
    
    for subset in ['type1_pro', 'type1_anti', 'type2_pro', 'type2_anti']:
        out_path = os.path.join(out_dir, f'winobias_{subset}.csv')
        if os.path.exists(out_path):
            continue
            
        logger.info(f"Downloading WinoBias {subset}...")
        try:
            dataset = load_dataset("wino_bias", subset, split="test")
            df = dataset.to_pandas()
            df.to_csv(out_path, index=False)
        except Exception as e:
            logger.error(f"Failed to download WinoBias {subset}: {e}")

def create_mlm_validation_set():
    """
    Extract 10,000 sentences from train_raw.jsonl to act as our fixed validation set.
    Removes them from train_raw to prevent leakage.
    """
    train_path = os.path.join(DATA_DIR, 'fineweb-edu', 'train_raw.jsonl')
    val_path = os.path.join(DATA_DIR, 'fineweb-edu', 'validation.jsonl')
    
    if os.path.exists(val_path):
        logger.info(f"Validation set already exists at {val_path}")
        return
        
    if not os.path.exists(train_path):
        logger.error(f"Training corpus not found at {train_path}. Cannot create validation set.")
        return
        
    logger.info("Creating fixed validation set (10,000 docs)...")
    
    val_docs = []
    temp_train_path = train_path + ".tmp"
    
    with open(train_path, 'r', encoding='utf-8') as f_in, \
         open(temp_train_path, 'w', encoding='utf-8') as f_out:
         
        for i, line in enumerate(f_in):
            if i < 10000:
                val_docs.append(line)
            else:
                f_out.write(line)
                
    with open(val_path, 'w', encoding='utf-8') as f_val:
        f_val.writelines(val_docs)
        
    os.replace(temp_train_path, train_path)
    logger.info(f"Saved {len(val_docs)} docs to {val_path} and removed them from train_raw.jsonl")

if __name__ == "__main__":
    logger.info("Starting evaluation datasets download...")
    download_multi_crows_pairs()
    download_indian_bias()
    download_winobias()
    create_mlm_validation_set()
    logger.info("Evaluation datasets download complete.")
