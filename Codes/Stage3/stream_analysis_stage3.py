import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime
from transformers import PreTrainedTokenizerFast
import argparse

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.architectures import build_model, get_model_info
from common.io_schemas import EARLY_MERGE_COLUMNS, STREAM_DISAGREEMENT_COLUMNS, TOKEN_DRIFT_COLUMNS
import Stage3.config_stage3 as cfg
from Stage1.eval_bias_stage1 import extract_model_metadata

logger = setup_logging('stream_analysis_stage3')

def simulate_early_merge(model, tokenizer, df, device, meta, out_path):
    """
    Simulate early merge by modifying CWSA. 
    Actually, we can't easily modify the architecture on the fly unless we rewrite the forward pass.
    Instead, we'll extract the streams and run the merge manually, then evaluate.
    Since EarlyMergeHyperloopBERT already exists, we could just evaluate that architecture vs HyperloopBERT.
    The spec says: "Compare HyperloopBERT vs EarlyMergeHyperloopBERT".
    So Early Merge Simulation is just comparing their bias summaries!
    We can do that in analyze_stage3.py.
    For this function, we just compute Early_Merge table if needed, or leave it to analyze_stage3.
    """
    pass

def analyze_stream_disagreement(model, tokenizer, df, device, meta, out_path):
    """
    Track the representations of each stream before CWSA.
    We compute the disagreement (variance or distance between streams) 
    and correlate it with effect size.
    """
    logger.info("Analyzing stream disagreement...")
    
    if meta['Architecture'] != 'HyperloopBERT':
        return
        
    results = []
    
    # Use built-in snapshot mechanism instead of hooks
    model.enable_stream_snapshots = True
    
    for idx, row in df.iterrows():
        stereo = row['stereo']
        anti = row['anti']
        
        try:
            inputs = tokenizer(stereo, return_tensors='pt', max_length=cfg.SEQ_LENGTH, truncation=True).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device if isinstance(device, str) else device.type,
                                    dtype=torch.bfloat16):
                    out = model(**inputs)
                    
            # stream_snapshots: dict[loop_idx -> list of (batch, seq_len, hidden) tensors]
            stream_snapshots = out.get('stream_snapshots', {})
            
            # Use final loop snapshot for disagreement (key = num_middle_loops)
            final_key = model.num_middle_loops
            if final_key not in stream_snapshots:
                # Fall back to last available key
                if not stream_snapshots:
                    continue
                final_key = max(stream_snapshots.keys())
                
            final_streams = stream_snapshots[final_key]  # list of (batch, seq_len, hidden) tensors
            
            # Extract CLS token (position 0) from each stream
            cls_streams = torch.stack([s[0, 0, :] for s in final_streams], dim=0)  # (num_streams, hidden)
            cls_streams = cls_streams.float()  # ensure fp32 for stable cosine computation
            
            # Compute pairwise cosine distances
            cos_sim = F.cosine_similarity(cls_streams.unsqueeze(1), cls_streams.unsqueeze(0), dim=-1)
            # Exclude self-similarity (diagonal = 1.0) when computing mean
            n = cos_sim.size(0)
            off_diag = cos_sim[~torch.eye(n, dtype=torch.bool, device=cos_sim.device)]
            disagreement = (1.0 - off_diag.mean()).item() if off_diag.numel() > 0 else 0.0
            
            from common.bias_metrics import score_bias_pair
            scores = score_bias_pair(model, tokenizer, stereo, anti, device)
            
            results.append({
                'Stage': cfg.STAGE,
                'Architecture': meta['Architecture'],
                'Model_Size': meta['Model_Size'],
                'Hidden_Size': model.hidden_size,
                'Seed': meta['Seed'],
                'Unique_Parameters': 0,
                'Total_Parameters': 0,
                'Effective_Depth': model.effective_depth,
                'Shared_Ratio': model.shared_ratio,
                'Loop_Depth': final_key,
                'Stream_Disagreement': disagreement,
                'Effect_Size': scores['Effect_Size'],
                'Pearson_R': None, 'Pearson_P': None, 'Spearman_R': None, 'Spearman_P': None,
                'Timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        except Exception as e:
            logger.debug(f"Row {idx} skipped: {e}")
            
    model.enable_stream_snapshots = False
            
    if results:
        res_df = pd.DataFrame(results)
        
        # Compute correlations
        from scipy.stats import pearsonr, spearmanr
        valid = res_df.dropna(subset=['Stream_Disagreement', 'Effect_Size'])
        if len(valid) > 2:
            pr, pp = pearsonr(valid['Stream_Disagreement'], valid['Effect_Size'])
            sr, sp = spearmanr(valid['Stream_Disagreement'], valid['Effect_Size'])
            res_df['Pearson_R'] = pr
            res_df['Pearson_P'] = pp
            res_df['Spearman_R'] = sr
            res_df['Spearman_P'] = sp
            
        agg_row = res_df.iloc[0].copy()
        agg_row['Stream_Disagreement'] = res_df['Stream_Disagreement'].mean()
        agg_row['Effect_Size'] = res_df['Effect_Size'].mean()
        
        out_df = pd.DataFrame([agg_row])
        for col in STREAM_DISAGREEMENT_COLUMNS:
            if col not in out_df.columns:
                out_df[col] = None
        out_df = out_df[STREAM_DISAGREEMENT_COLUMNS]
        
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        mode = 'a' if os.path.exists(out_path) else 'w'
        header = not os.path.exists(out_path)
        out_df.to_csv(out_path, mode=mode, header=header, index=False)

def analyze_token_drift(model, tokenizer, df, device, meta, out_path):
    """
    Map drift of demographic vs stereotypical tokens across loops.
    """
    logger.info("Analyzing token drift...")
    # Simplified version: similar to loop trajectory, track cosine similarity 
    # of representations at different depths for specific tokens.
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage3.py directly instead.")
        return
        
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, cfg.RESULTS_DIR)
    models_dir = os.path.join(base_dir, cfg.MODELS_DIR)
    
    tokenizer_dir = os.path.join(data_dir, 'tokenizer')
    if not os.path.exists(tokenizer_dir):
        return
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    eval_dir = os.path.join(data_dir, 'datasets_eval')
    multicrows_path = os.path.join(eval_dir, 'multicrows', 'crows_pair_english.csv')
    if not os.path.exists(multicrows_path):
        return
        
    df = pd.read_csv(multicrows_path)
    if len(df) > 100:
        df = df.sample(n=100, random_state=42)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoints = []
    type_dir = os.path.join(models_dir, 'iso_band_models')
    if os.path.exists(type_dir):
        for root, dirs, files in os.walk(type_dir):
            if 'pytorch_model.bin' in files:
                checkpoints.append(root)
                
    disagreement_out = os.path.join(results_dir, 'mechanistic', 'stream_disagreement.csv')
    
    for cp_dir in checkpoints:
        meta = extract_model_metadata(cp_dir)
        if not meta or meta['Architecture'] != 'HyperloopBERT':
            continue
            
        num_streams = cfg.DEFAULT_NUM_STREAMS
        if 'streams' in meta['Run_ID']:
            parts = meta['Run_ID'].split('_')
            for p in parts:
                if p.startswith('streams'):
                    try:
                        num_streams = int(p.replace('streams', ''))
                    except:
                        pass
        
        if num_streams != cfg.DEFAULT_NUM_STREAMS:
            continue
            
        model = build_model(meta['Architecture'], meta['Model_Size'], num_streams=num_streams)
        try:
            model.load_state_dict(torch.load(os.path.join(cp_dir, 'pytorch_model.bin'), map_location='cpu', weights_only=True))
        except:
            continue
            
        model.to(device)
        model.eval()
        
        analyze_stream_disagreement(model, tokenizer, df, device, meta, disagreement_out)
        
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
