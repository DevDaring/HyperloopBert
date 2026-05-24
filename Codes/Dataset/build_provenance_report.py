import os
import sys
import json
import hashlib
from datetime import datetime
import pandas as pd

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging

logger = setup_logging('build_provenance_report')

def _calculate_md5(filepath: str) -> str:
    """Calculate MD5 hash of a file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
        
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def _count_rows(filepath: str) -> int:
    """Count logical rows (documents or pairs)."""
    if not os.path.exists(filepath):
        return 0
        
    if filepath.endswith('.csv'):
        try:
            df = pd.read_csv(filepath)
            return len(df)
        except:
            return 0
    elif filepath.endswith('.jsonl'):
        count = 0
        with open(filepath, 'rb') as f:
            for _ in f:
                count += 1
        return count
    return 0

def build_report(data_dir: str, output_path: str):
    """
    Build a provenance report for all datasets.
    """
    logger.info(f"Building provenance report for {data_dir}...")
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    report_lines = [
        "# HyperloopBert Data Provenance Report",
        "",
        f"**Generated:** {timestamp}",
        "",
        "This report documents the exact state of all datasets used in the pipeline.",
        "Any changes to these files will break reproducibility.",
        "",
        "## Training Corpus",
        ""
    ]
    
    train_raw = os.path.join(data_dir, 'fineweb-edu', 'train_raw.jsonl')
    train_filtered = os.path.join(data_dir, 'fineweb-edu', 'train_filtered.jsonl')
    val_set = os.path.join(data_dir, 'fineweb-edu', 'validation.jsonl')
    
    for name, path in [("FineWeb-Edu (Raw)", train_raw), 
                       ("FineWeb-Edu (Filtered)", train_filtered),
                       ("Validation Set (Holdout)", val_set)]:
        rows = _count_rows(path)
        md5 = _calculate_md5(path)
        rel_path = os.path.relpath(path, start=os.path.dirname(output_path))
        report_lines.extend([
            f"### {name}",
            f"- **Path:** `{rel_path}`",
            f"- **Rows:** {rows:,}",
            f"- **MD5:** `{md5}`",
            ""
        ])
        
    report_lines.extend([
        "## Evaluation Datasets",
        ""
    ])
    
    eval_dir = os.path.join(data_dir, 'datasets_eval')
    if os.path.exists(eval_dir):
        for root, _, files in os.walk(eval_dir):
            for file in files:
                if file.endswith('.csv'):
                    path = os.path.join(root, file)
                    rel_path = os.path.relpath(path, start=os.path.dirname(output_path))
                    rows = _count_rows(path)
                    md5 = _calculate_md5(path)
                    
                    dataset_name = os.path.basename(os.path.dirname(path))
                    
                    report_lines.extend([
                        f"### {dataset_name} / {file}",
                        f"- **Path:** `{rel_path}`",
                        f"- **Rows:** {rows:,}",
                        f"- **MD5:** `{md5}`",
                        ""
                    ])
                    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    logger.info(f"Provenance report written to {output_path}")

if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    REPORT_PATH = os.path.join(DATA_DIR, 'provenance_report.md')
    build_report(DATA_DIR, REPORT_PATH)
