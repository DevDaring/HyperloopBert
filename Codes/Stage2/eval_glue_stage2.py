import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datetime import datetime
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from tqdm import tqdm
import argparse

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.architectures import build_model, get_model_info, BertPooler
from common.bias_metrics import passes_quality_screen
from common.io_schemas import GLUE_SUMMARY_COLUMNS
import Stage2.config_stage2 as cfg
from Stage1.eval_bias_stage1 import extract_model_metadata, get_snapshot_mlm_quality

logger = setup_logging('eval_glue_stage2')

class SequenceClassificationModel(nn.Module):
    def __init__(self, encoder, hidden_size, num_labels=2):
        super().__init__()
        self.encoder = encoder
        self.pooler = BertPooler(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use last hidden state for pooling
        hidden_states = outputs.get('last_hidden_state', outputs[0])
        pooled_output = self.pooler(hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train_and_eval_glue(model_path, meta, model_info, task, tokenizer, device):
    """Fine-tune model on GLUE task and return eval metrics."""
    logger.info(f"Fine-tuning {meta['Architecture']} on GLUE/{task}...")
    
    # Load dataset
    try:
        dataset = load_dataset('glue', task)
    except Exception as e:
        logger.error(f"Failed to load GLUE task {task}: {e}")
        return {'Accuracy': 0.0, 'F1': 0.0}
        
    is_regression = task == 'stsb'
    num_labels = 1 if is_regression else len(dataset['train'].features['label'].names)
    
    # Tokenize
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    
    key1, key2 = task_to_keys[task]
    
    def tokenize_function(examples):
        if key2 is None:
            return tokenizer(examples[key1], padding="max_length", truncation=True, max_length=cfg.SEQ_LENGTH)
        return tokenizer(examples[key1], examples[key2], padding="max_length", truncation=True, max_length=cfg.SEQ_LENGTH)
        
    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    train_dataloader = DataLoader(encoded_dataset['train'], batch_size=cfg.GLUE_BATCH_SIZE, shuffle=True)
    eval_split = 'validation_mismatched' if task == 'mnli' else 'validation'
    eval_dataloader = DataLoader(encoded_dataset[eval_split], batch_size=cfg.GLUE_BATCH_SIZE)
    
    # Build model
    encoder = build_model(meta['Architecture'], meta['Model_Size'])
    try:
        encoder.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    except Exception as e:
        logger.error(f"Failed to load encoder from {model_path}: {e}")
        return {'Accuracy': 0.0, 'F1': 0.0}
        
    model = SequenceClassificationModel(encoder, model_info['Hidden_Size'], num_labels)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.GLUE_LR)
    
    if is_regression:
        loss_fct = nn.MSELoss()
    else:
        loss_fct = nn.CrossEntropyLoss()
        
    # Training Loop
    for epoch in range(cfg.GLUE_EPOCHS):
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Support FP16/BF16 if desired, using FP32 for GLUE fine-tuning stability here
            logits = model(input_ids, attention_mask)
            
            if is_regression:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            logits = model(input_ids, attention_mask)
            
            if is_regression:
                preds = logits.squeeze().cpu()
            else:
                preds = torch.argmax(logits, dim=-1).cpu()
                
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate metrics
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    
    if is_regression:
        # For STS-B, report Pearson/Spearman as 'Accuracy'/'F1' just to fit schema, 
        # but Stage 2 only needs SST-2 and RTE (classification).
        return {'Accuracy': 0.0, 'F1': 0.0}
    else:
        acc = accuracy_score(all_labels, all_preds)
        try:
            f1 = f1_score(all_labels, all_preds, average='macro')
        except:
            f1 = 0.0
            
    logger.info(f"Task {task} -> Acc: {acc:.4f}, F1: {f1:.4f}")
    return {'Accuracy': float(acc), 'F1': float(f1)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
    parser.add_argument('--resume', action='store_true', help='Resume existing eval')
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage2.py directly instead.")
        return
        
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, cfg.RESULTS_DIR)
    models_dir = os.path.join(base_dir, cfg.MODELS_DIR)
    
    tokenizer_dir = os.path.join(data_dir, 'tokenizer')
    if not os.path.exists(tokenizer_dir):
        logger.error(f"Tokenizer not found at {tokenizer_dir}")
        return
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoints = []
    # GLUE eval typically done ONLY on primary iso-loss bands to save compute
    # Here we'll process iso_band_models
    type_dir = os.path.join(models_dir, 'iso_band_models')
    if os.path.exists(type_dir):
        for root, dirs, files in os.walk(type_dir):
            if 'pytorch_model.bin' in files:
                checkpoints.append(root)
                
    summary_path = os.path.join(results_dir, 'glue', 'summary_table.csv')
    
    for cp_dir in checkpoints:
        meta = extract_model_metadata(cp_dir)
        if not meta:
            continue
            
        pp = get_snapshot_mlm_quality(results_dir, meta['Architecture'], meta['Model_Size'], 
                                      meta['Seed'], meta['Band'], meta['Token_Marker'])
                                      
        if not passes_quality_screen(pp, cfg.PSEUDO_PERPLEXITY_QUALITY_THRESHOLD):
            continue
            
        # Check resume
        if args.resume and os.path.exists(summary_path):
            sum_df = pd.read_csv(summary_path)
            mask = (sum_df['Architecture'] == meta['Architecture']) & \
                   (sum_df['Model_Size'] == meta['Model_Size']) & \
                   (sum_df['Seed'] == meta['Seed']) & \
                   (sum_df['Band'] == meta['Band'])
            
            # If we have rows for all required tasks, skip
            existing_tasks = sum_df[mask]['Task'].unique() if not sum_df[mask].empty else []
            if set(cfg.GLUE_TASKS).issubset(set(existing_tasks)):
                logger.info(f"Skipping GLUE for {cp_dir} - already complete.")
                continue
                
        # We need model info (hidden size) without instantiating the full model here, 
        # but build_model handles that
        temp_model = build_model(meta['Architecture'], meta['Model_Size'])
        model_info = get_model_info(temp_model)
        del temp_model
        
        task_metrics = {}
        for task in cfg.GLUE_TASKS:
            model_path = os.path.join(cp_dir, 'pytorch_model.bin')
            metrics = train_and_eval_glue(model_path, meta, model_info, task, tokenizer, device)
            task_metrics[task] = metrics
            
            # Compute partial average if possible
            glue_avg = sum(m['Accuracy'] for m in task_metrics.values()) / len(task_metrics)
            
            res_row = {
                'Stage': cfg.STAGE,
                'Architecture': meta['Architecture'],
                'Model_Size': meta['Model_Size'],
                'Hidden_Size': model_info['Hidden_Size'],
                'Seed': meta['Seed'],
                'Unique_Parameters': model_info['Unique_Parameters'],
                'Total_Parameters': model_info['Total_Parameters'],
                'Effective_Depth': model_info['Effective_Depth'],
                'Shared_Ratio': model_info['Shared_Ratio'],
                'Task': task,
                'Accuracy': metrics['Accuracy'],
                'F1': metrics['F1'],
                'GLUE_Average': glue_avg, # Note: true GLUE avg needs all tasks
                'Timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
            df_sum = pd.DataFrame([res_row])[GLUE_SUMMARY_COLUMNS]
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            mode = 'a' if os.path.exists(summary_path) else 'w'
            header = not os.path.exists(summary_path)
            df_sum.to_csv(summary_path, mode=mode, header=header, index=False)

if __name__ == "__main__":
    main()
