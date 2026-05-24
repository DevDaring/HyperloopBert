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
from common.io_schemas import BIAS_TRAJECTORY_COLUMNS
import Stage2.config_stage2 as cfg
from Stage1.eval_bias_stage1 import extract_model_metadata

logger = setup_logging('loop_trajectory_stage2')

def extract_intermediate_representations(model, input_ids, attention_mask, architecture):
    """
    Run forward pass and extract hidden states at each loop boundary.
    For VanillaBERT, we sample at equivalent depths (3, 6, 9, 12).
    For LoopedBERT/ALBERT, we sample after each loop iteration.
    """
    reps = {} # {depth: tensor}
    
    # We use a hook-based approach. We register hooks on the specific layers we want.
    handles = []
    
    if architecture == 'VanillaBERT':
        # Layers 3, 6, 9, 12 (0-indexed: 2, 5, 8, 11)
        target_layers = {2: 3, 5: 6, 8: 9, 11: 12}
        
        def get_hook(depth):
            def hook(module, inp, output):
                reps[depth] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
            return hook
            
        for idx, depth in target_layers.items():
            handles.append(model.encoder.layer[idx].register_forward_hook(get_hook(depth)))
            
    elif architecture == 'LoopedBERT':
        # Depth markers: end of begin block (2), then after each middle loop iteration (4, 6, 8, 10), then end (12)
        # However, middle block is applied iteratively. We need to hook inside the forward of the model.
        # Given architectures.py design, we might not be able to easily hook a specific iteration 
        # of the same layer object via standard PyTorch hooks because the hook fires every time.
        # We can use a counter in the hook.
        
        call_count = [0]
        
        def middle_hook(module, inp, output):
            call_count[0] += 1
            # middle block is 2 layers. It gets called 4 times.
            # depth = 2 (begin) + call_count * 2
            depth = 2 + call_count[0] * 2
            reps[depth] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
            
        handles.append(model.encoder.middle_block[-1].register_forward_hook(middle_hook))
        
        def end_hook(module, inp, output):
            reps[12] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
            
        handles.append(model.encoder.end_block[-1].register_forward_hook(end_hook))
        
    elif architecture == 'ALBERTLoopedBERT':
        call_count = [0]
        
        def shared_hook(module, inp, output):
            call_count[0] += 1
            depth = call_count[0]
            reps[depth] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
            
        handles.append(model.encoder.shared_layer.register_forward_hook(shared_hook))
        
    # Run forward
    with torch.no_grad():
        with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16):
            model(input_ids=input_ids, attention_mask=attention_mask)
            
    # Cleanup
    for h in handles:
        h.remove()
        
    return reps

def classify_trajectory(rates):
    """
    rates: list of preference rates ordered by depth.
    Classify as CONVERGENT, AMPLIFYING, or OSCILLATING.
    """
    if len(rates) < 2:
        return "UNKNOWN"
        
    diffs = [rates[i] - rates[i-1] for i in range(1, len(rates))]
    
    if all(d <= 0.01 for d in diffs) and rates[0] > rates[-1]:
        return "CONVERGENT"
    elif all(d >= -0.01 for d in diffs) and rates[-1] > rates[0]:
        return "AMPLIFYING"
    else:
        return "OSCILLATING"

def run_trajectory_analysis(model, tokenizer, df, device, meta, out_path):
    """Run loop trajectory analysis and save to CSV."""
    logger.info(f"Running trajectory analysis for {meta['Architecture']} {meta['Model_Size']} Seed {meta['Seed']}")
    
    dataset_name = 'Multi-CrowS-Pairs' # Defaulting for Stage 2 trajectory
    
    # We will score a subset (e.g. 100 pairs) to keep it fast, or the full dataset if small
    # For now, process all valid rows.
    
    depth_scores = {} # depth -> [{'stereo_pll', 'anti_pll', 'pref', 'effect'}]
    
    for idx, row in df.iterrows():
        stereo = row['stereo']
        anti = row['anti']
        category = row.get('bias_type', row.get('category', 'unknown'))
        
        stereo_inputs = tokenizer(stereo, return_tensors="pt", max_length=128, truncation=True).to(device)
        anti_inputs = tokenizer(anti, return_tensors="pt", max_length=128, truncation=True).to(device)
        
        stereo_reps = extract_intermediate_representations(model, stereo_inputs['input_ids'], stereo_inputs['attention_mask'], meta['Architecture'])
        anti_reps = extract_intermediate_representations(model, anti_inputs['input_ids'], anti_inputs['attention_mask'], meta['Architecture'])
        
        # Determine shared indices for SS-PLL style evaluation at intermediate depths
        # For simplicity, we just do full PLL here, as defined in metric schema.
        # But we need to apply MLM head to intermediate reps.
        
        for depth in stereo_reps.keys():
            if depth not in anti_reps:
                continue
                
            # Apply MLM head
            with torch.no_grad():
                with torch.autocast(device_type=device if isinstance(device, str) else device.type, dtype=torch.bfloat16):
                    stereo_logits = model.mlm_head(stereo_reps[depth])
                    anti_logits = model.mlm_head(anti_reps[depth])
                    
            # Compute PLL using the logits
            def pll_from_logits(logits, input_ids):
                seq_len = input_ids.size(1)
                total_log_prob = 0.0
                num_tokens = seq_len - 2
                if num_tokens <= 0: return 0.0
                
                # We approximate by taking the probability of the actual token 
                # given the unmasked context up to this depth. This is not true MLM PLL
                # since inputs weren't masked, but it serves as a proxy for how the representation
                # encodes the tokens at intermediate layers (next-token or self-reconstruction prob).
                # Actually, standard approach: hook into masked passes.
                # Since we want actual PLL, we would need to run the full O(N) masking pass
                # and hook EVERY time, which is O(N * Depth * Pairs) - extremely slow.
                # To approximate: we just compute the sum of log probs of the input tokens
                # given the unmasked forward pass (pseudo-reconstruction loss).
                
                log_probs = F.log_softmax(logits, dim=-1)
                for i in range(1, seq_len - 1):
                    token_id = input_ids[0, i]
                    total_log_prob += log_probs[0, i, token_id].item()
                return total_log_prob / num_tokens
                
            pll_stereo = pll_from_logits(stereo_logits, stereo_inputs['input_ids'])
            pll_anti = pll_from_logits(anti_logits, anti_inputs['input_ids'])
            
            effect = pll_stereo - pll_anti
            pref = 1 if pll_stereo > pll_anti else 0
            
            if depth not in depth_scores:
                depth_scores[depth] = []
                
            depth_scores[depth].append({
                'Category': category,
                'Pref': pref,
                'Effect': effect
            })
            
    # Aggregate
    results = []
    
    # We also compute trajectory shape per category
    # First, build a map of category -> depth -> mean_pref
    cat_depth_prefs = {}
    for depth, scores in depth_scores.items():
        df_scores = pd.DataFrame(scores)
        cat_means = df_scores.groupby('Category')['Pref'].mean().to_dict()
        for cat, mean_pref in cat_means.items():
            if cat not in cat_depth_prefs:
                cat_depth_prefs[cat] = {}
            cat_depth_prefs[cat][depth] = mean_pref
            
    for depth, scores in depth_scores.items():
        df_scores = pd.DataFrame(scores)
        
        # Overall
        mean_pref = df_scores['Pref'].mean()
        std_pref = df_scores['Pref'].std() # Approx
        mean_effect = df_scores['Effect'].mean()
        
        # Determine overall shape
        # Requires looking across all depths, so we do it after
        
        res_row = {
            'Stage': cfg.STAGE,
            'Architecture': meta['Architecture'],
            'Model_Size': meta['Model_Size'],
            'Seed': meta['Seed'],
            'Dataset': dataset_name,
            'Category': 'ALL',
            'Loop_Depth': depth,
            'Mean_Preference_Rate': mean_pref,
            'Std_Preference_Rate': std_pref,
            'Mean_Effect_Size': mean_effect,
            'Trajectory_Shape': 'UNKNOWN', # Filled later
            'Timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        results.append(res_row)
        
        # Per category
        for cat, group in df_scores.groupby('Category'):
            results.append({
                'Stage': cfg.STAGE,
                'Architecture': meta['Architecture'],
                'Model_Size': meta['Model_Size'],
                'Seed': meta['Seed'],
                'Dataset': dataset_name,
                'Category': cat,
                'Loop_Depth': depth,
                'Mean_Preference_Rate': group['Pref'].mean(),
                'Std_Preference_Rate': group['Pref'].std(),
                'Mean_Effect_Size': group['Effect'].mean(),
                'Trajectory_Shape': 'UNKNOWN',
                'Timestamp': datetime.utcnow().isoformat() + 'Z'
            })
            
    res_df = pd.DataFrame(results)
    
    # Compute Trajectory Shape
    for cat in res_df['Category'].unique():
        cat_df = res_df[res_df['Category'] == cat].sort_values('Loop_Depth')
        rates = cat_df['Mean_Preference_Rate'].tolist()
        shape = classify_trajectory(rates)
        res_df.loc[res_df['Category'] == cat, 'Trajectory_Shape'] = shape
        
    res_df = res_df[BIAS_TRAJECTORY_COLUMNS]
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mode = 'a' if os.path.exists(out_path) else 'w'
    header = not os.path.exists(out_path)
    res_df.to_csv(out_path, mode=mode, header=header, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Delegate to Dry_Run')
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
        return
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    # Only run on Multi-CrowS-Pairs to save compute
    eval_dir = os.path.join(data_dir, 'datasets_eval')
    multicrows_path = os.path.join(eval_dir, 'multicrows', 'crows_pair_english.csv')
    if not os.path.exists(multicrows_path):
        return
        
    df = pd.read_csv(multicrows_path)
    # Take a sample for trajectory analysis to save time (e.g., 200 pairs)
    if len(df) > 200:
        df = df.sample(n=200, random_state=42)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoints = []
    # Use primary band models
    type_dir = os.path.join(models_dir, 'iso_band_models')
    if os.path.exists(type_dir):
        for root, dirs, files in os.walk(type_dir):
            if 'pytorch_model.bin' in files:
                checkpoints.append(root)
                
    out_path = os.path.join(results_dir, 'mechanistic', 'loop_trajectory.csv')
    
    for cp_dir in checkpoints:
        meta = extract_model_metadata(cp_dir)
        if not meta:
            continue
            
        # Optional: check if we already processed this model
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            mask = (existing['Architecture'] == meta['Architecture']) & \
                   (existing['Model_Size'] == meta['Model_Size']) & \
                   (existing['Seed'] == meta['Seed'])
            if not existing[mask].empty:
                continue
                
        model = build_model(meta['Architecture'], meta['Model_Size'])
        try:
            model.load_state_dict(torch.load(os.path.join(cp_dir, 'pytorch_model.bin'), map_location='cpu', weights_only=True))
        except:
            continue
            
        model.to(device)
        model.eval()
        
        run_trajectory_analysis(model, tokenizer, df, device, meta, out_path)
        
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
