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
from common.attention import force_full_precision_attention
from common.io_schemas import BIAS_TRAJECTORY_COLUMNS
import Stage2.config_stage2 as cfg
from Stage1.eval_bias_stage1 import extract_model_metadata

logger = setup_logging('loop_trajectory_stage2')

def extract_intermediate_representations(model, input_ids, attention_mask, architecture):
    """
    Run ONE forward pass and capture hidden states at matched effective depths.

    Depth convention (effective layer applications, comparable across models):
      VanillaBERT       : after layers 3, 6, 9, 12  (encoder is an nn.ModuleList)
      LoopedBERT        : after begin (2), each loop iteration (4, 6, 8, 10),
                          and the end block (12)
      ALBERTLoopedBERT  : after each of the 12 shared-layer applications
      HyperloopBERT     : after begin (2), each loop's shared block (4,6,8,10),
                          and the end block (12) -- block output is the
                          stream-mixed representation

    Since shared modules fire multiple times per forward, hooks use a call
    counter that is reset before the forward pass.
    """
    reps = {}  # {depth: tensor (batch, seq, hidden)}
    handles = []

    def _out(output):
        t = output[0] if isinstance(output, tuple) else output
        return t.detach().clone()

    if architecture == 'VanillaBERT':
        # model.encoder is an nn.ModuleList of 12 BertLayers
        target_layers = {2: 3, 5: 6, 8: 9, 11: 12}

        def get_hook(depth):
            def hook(module, inp, output):
                reps[depth] = _out(output)
            return hook

        for idx, depth in target_layers.items():
            handles.append(model.encoder[idx].register_forward_hook(get_hook(depth)))

    elif architecture in ('LoopedBERT', 'HyperloopBERT'):
        call_count = [0]

        def begin_hook(module, inp, output):
            reps[2] = _out(output)

        def middle_hook(module, inp, output):
            call_count[0] += 1
            depth = 2 + call_count[0] * 2  # 4, 6, 8, 10
            reps[depth] = _out(output)

        def end_hook(module, inp, output):
            reps[12] = _out(output)

        handles.append(model.begin_layers[-1].register_forward_hook(begin_hook))
        handles.append(model.middle_layers[-1].register_forward_hook(middle_hook))
        handles.append(model.end_layers[-1].register_forward_hook(end_hook))

    elif architecture == 'ALBERTLoopedBERT':
        call_count = [0]

        def shared_hook(module, inp, output):
            call_count[0] += 1
            reps[call_count[0]] = _out(output)

        handles.append(model.shared_layer.register_forward_hook(shared_hook))

    else:
        return reps

    try:
        with torch.no_grad():
            # FP32 forward (pre-registered scoring precision): PLL gaps on
            # borderline pairs are the same order as BF16 rounding, so the
            # trajectory probe must match the primary scorer's precision.
            with force_full_precision_attention(), \
                 torch.autocast(device_type=input_ids.device.type, enabled=False):
                model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    return reps


def _modified_token_indices(ids_a, ids_b):
    """
    Indices in sequence A that fall OUTSIDE the common prefix/suffix shared
    with sequence B (i.e. the tokens the stereotype manipulation changed),
    excluding [CLS]/[SEP].
    """
    prefix_len = 0
    while prefix_len < min(len(ids_a), len(ids_b)) and ids_a[prefix_len] == ids_b[prefix_len]:
        prefix_len += 1
    suffix_len = 0
    while suffix_len < min(len(ids_a) - prefix_len, len(ids_b) - prefix_len) and \
          ids_a[len(ids_a) - 1 - suffix_len] == ids_b[len(ids_b) - 1 - suffix_len]:
        suffix_len += 1
    indices = [i for i in range(prefix_len, len(ids_a) - suffix_len)
               if 0 < i < len(ids_a) - 1]
    return indices


def masked_pll_at_depths(model, tokenizer, sentence, other_sentence, device, max_length=128):
    """
    True masked PLL of the MODIFIED tokens, computed at every hooked depth.

    For each token position the pair manipulation changed, that token is
    replaced with [MASK]; all masked variants are scored in one batched
    forward pass with depth hooks; the MLM head is applied to each depth's
    hidden state at the masked position.

    Returns dict {depth: mean log-prob of the true tokens} or None if the
    pair has no scoreable modified tokens.
    """
    enc = tokenizer(sentence, return_tensors='pt', max_length=max_length, truncation=True)
    enc_other = tokenizer(other_sentence, return_tensors='pt', max_length=max_length, truncation=True)
    ids = enc['input_ids'][0].tolist()
    other_ids = enc_other['input_ids'][0].tolist()

    mod_indices = _modified_token_indices(ids, other_ids)
    if not mod_indices:
        return None

    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    masked_batch = input_ids.repeat(len(mod_indices), 1)
    for row, i in enumerate(mod_indices):
        masked_batch[row, i] = tokenizer.mask_token_id
    masked_attn = attention_mask.repeat(len(mod_indices), 1)

    arch = type(model).__name__
    reps = extract_intermediate_representations(model, masked_batch, masked_attn, arch)
    if not reps:
        return None

    depth_pll = {}
    with torch.no_grad():
        for depth, rep in reps.items():
            with torch.autocast(device_type=masked_batch.device.type, enabled=False):
                logits = model.mlm_head(rep.to(device).float())
            total = 0.0
            for row, i in enumerate(mod_indices):
                log_probs = F.log_softmax(logits[row, i, :].float(), dim=-1)
                total += log_probs[input_ids[0, i]].item()
            depth_pll[depth] = total / len(mod_indices)
    return depth_pll

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

    # Identity columns (Hidden_Size/Unique_Parameters/Total_Parameters/
    # Effective_Depth/Shared_Ratio) are required by BIAS_TRAJECTORY_COLUMNS but
    # are not in `meta` -- read them from the model itself.
    model_info = get_model_info(model)

    dataset_name = 'Multi-CrowS-Pairs' # Defaulting for Stage 2 trajectory
    
    # We will score a subset (e.g. 100 pairs) to keep it fast, or the full dataset if small
    # For now, process all valid rows.
    
    depth_scores = {} # depth -> [{'stereo_pll', 'anti_pll', 'pref', 'effect'}]
    
    # True masked PLL of the MODIFIED tokens at every depth: each modified
    # token is masked and predicted, depth activations are captured with
    # forward hooks in a single batched pass per sentence. This is the
    # standard CrowS-Pairs quantity restricted to the manipulated span,
    # evaluated at intermediate effective depths.
    for idx, row in df.iterrows():
        stereo = row['stereo']
        anti = row['anti']
        category = row.get('bias_type', row.get('category', 'unknown'))

        stereo_depth_pll = masked_pll_at_depths(model, tokenizer, stereo, anti, device)
        anti_depth_pll = masked_pll_at_depths(model, tokenizer, anti, stereo, device)
        if not stereo_depth_pll or not anti_depth_pll:
            continue

        for depth in stereo_depth_pll:
            if depth not in anti_depth_pll:
                continue
            pll_stereo = stereo_depth_pll[depth]
            pll_anti = anti_depth_pll[depth]

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
            'Hidden_Size': model_info['Hidden_Size'],
            'Seed': meta['Seed'],
            'Unique_Parameters': model_info['Unique_Parameters'],
            'Total_Parameters': model_info['Total_Parameters'],
            'Effective_Depth': model_info['Effective_Depth'],
            'Shared_Ratio': model_info['Shared_Ratio'],
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
                'Hidden_Size': model_info['Hidden_Size'],
                'Seed': meta['Seed'],
                'Unique_Parameters': model_info['Unique_Parameters'],
                'Total_Parameters': model_info['Total_Parameters'],
                'Effective_Depth': model_info['Effective_Depth'],
                'Shared_Ratio': model_info['Shared_Ratio'],
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
    parser.add_argument('--stage', type=int, default=2, choices=[2, 3],
                        help='Which stage models/results to analyse (3 reuses this script)')
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Dry run flag detected. Use python Dry_Run/dry_run_stage2.py directly instead.")
        return

    global cfg
    if args.stage == 3:
        import Stage3.config_stage3 as cfg3
        cfg = cfg3
        logger.info("Loop trajectory running against STAGE 3 models/results.")

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
    
    # Snapshot selection preserves iso-loss matching: use the COMMON primary
    # band per size (the band every architecture crossed for every shared
    # seed), NOT each architecture's own deepest band -- mixing bands would
    # re-introduce the quality confound into the trajectory comparison.
    from common.iso_loss import compute_primary_band
    mlm_path = os.path.join(results_dir, 'mlm', 'summary_table.csv')
    mlm_df = pd.read_csv(mlm_path) if os.path.exists(mlm_path) else None
    primary_band_by_size = {}
    for size in getattr(cfg, 'SIZES', ['base']):
        primary_band_by_size[size] = compute_primary_band(
            mlm_df, cfg.ARCHITECTURES, size, logger_obj=logger)
        logger.info(f"Trajectory primary band ({size}): {primary_band_by_size[size]}")

    candidates = {}
    type_dir = os.path.join(models_dir, 'iso_band_models')
    if os.path.exists(type_dir):
        for root, dirs, files in os.walk(type_dir):
            if 'pytorch_model.bin' not in files:
                continue
            meta = extract_model_metadata(root)
            if not meta or meta['Band'] is None:
                continue
            if meta.get('Merge_At') is not None:
                continue
            if meta.get('Stream_Count') not in (None, 4):
                continue  # trajectory uses the primary arms only
            target = primary_band_by_size.get(meta['Model_Size'])
            if target is None or meta['Band'] != target:
                continue
            key = (meta['Architecture'], meta['Model_Size'], meta['Seed'])
            candidates[key] = (meta['Band'], root)
    checkpoints = [root for _, root in candidates.values()]
    if not checkpoints:
        logger.warning("No snapshots at the common primary band; trajectory not run.")

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
