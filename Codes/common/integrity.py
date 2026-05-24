import os
import json
import hashlib
import csv
from datetime import datetime
from typing import Dict, List, Optional
import shutil

# NOTE: Integrity checks run at the start of EVERY rerun before consuming any data.
#       This detects duplicate and corrupted records and prevents silent data quality issues.

def _hash_text(text: str) -> str:
    """Hash normalized text to detect exact duplicates."""
    normalized = text.strip().lower()
    return hashlib.sha1(normalized.encode('utf-8')).hexdigest()

def check_jsonl_corruption(filepath: str, quarantine_path: Optional[str] = None, min_text_length: int = 10) -> Dict:
    """
    Check a JSONL file for corruption:
    - Each line must parse as valid JSON
    - 'text' field must be present, non-empty, and >= min_text_length chars
    - Valid UTF-8
    - No truncated final line
    Bad lines are moved to quarantine_path if specified.
    Returns dict: total_lines, corrupt_count, quarantined_count
    """
    if not os.path.exists(filepath):
        return {'total_lines': 0, 'corrupt_count': 0, 'quarantined_count': 0}
        
    total_lines = 0
    corrupt_count = 0
    quarantined_count = 0
    
    valid_lines = []
    quarantine_lines = []
    
    with open(filepath, 'rb') as f:
        for raw_line in f:
            total_lines += 1
            is_valid = True
            
            try:
                line = raw_line.decode('utf-8')
                if not line.endswith('\n') and total_lines > 1:
                    # Truncated final line is okay if it's the last, but let's check parseability
                    pass
                    
                obj = json.loads(line)
                if 'text' not in obj or not isinstance(obj['text'], str) or len(obj['text'].strip()) < min_text_length:
                    is_valid = False
            except (UnicodeDecodeError, json.JSONDecodeError):
                is_valid = False
                
            if is_valid:
                valid_lines.append(raw_line)
            else:
                corrupt_count += 1
                if quarantine_path:
                    quarantine_lines.append(raw_line)
                    quarantined_count += 1
                    
    if corrupt_count > 0:
        # Overwrite file with only valid lines
        with open(filepath, 'wb') as f:
            f.writelines(valid_lines)
            
        if quarantine_path and quarantine_lines:
            os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)
            with open(quarantine_path, 'ab') as f:
                f.writelines(quarantine_lines)
                
    return {
        'total_lines': total_lines,
        'corrupt_count': corrupt_count,
        'quarantined_count': quarantined_count
    }

def deduplicate_jsonl(filepath: str, output_path: Optional[str] = None, hash_field: str = 'text') -> Dict:
    """
    Detect and remove exact duplicates from a JSONL file.
    Duplicate detection: SHA-1 of normalized (stripped, lowercased) text.
    If output_path is None, modifies in place (with backup).
    Returns dict: total_lines, duplicate_count, unique_count, duplicate_rate
    """
    if not os.path.exists(filepath):
        return {'total_lines': 0, 'duplicate_count': 0, 'unique_count': 0, 'duplicate_rate': 0.0}
        
    seen_hashes = set()
    total_lines = 0
    unique_lines = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                obj = json.loads(line)
                if hash_field in obj and isinstance(obj[hash_field], str):
                    h = _hash_text(obj[hash_field])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        unique_lines.append(line)
                else:
                    unique_lines.append(line) # Keep lines without the field to be safe
            except json.JSONDecodeError:
                unique_lines.append(line) # Keep corrupt lines, check_jsonl_corruption handles them
                
    duplicate_count = total_lines - len(unique_lines)
    unique_count = len(unique_lines)
    duplicate_rate = duplicate_count / total_lines if total_lines > 0 else 0.0
    
    if duplicate_count > 0:
        out_path = output_path if output_path else filepath
        if out_path == filepath:
            backup_path = filepath + ".bak"
            shutil.copy2(filepath, backup_path)
            
        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(unique_lines)
            
    return {
        'total_lines': total_lines,
        'duplicate_count': duplicate_count,
        'unique_count': unique_count,
        'duplicate_rate': duplicate_rate
    }

def _calculate_md5(filepath: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def _count_rows(filepath: str) -> int:
    """Count rows in a text file (JSONL or CSV)."""
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    # For CSVs, subtract 1 for header if it's a CSV file
    if filepath.endswith('.csv') and count > 0:
        count -= 1
    return count

def build_manifest(file_paths: List[str], manifest_path: str) -> Dict:
    """
    Build a JSON manifest recording: filepath, row_count, md5_hash, timestamp.
    Writes to manifest_path.
    Returns: dict manifest
    """
    manifest = {}
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    for path in file_paths:
        if os.path.exists(path):
            rel_path = os.path.relpath(path, start=os.path.dirname(manifest_path))
            manifest[rel_path] = {
                'row_count': _count_rows(path),
                'md5_hash': _calculate_md5(path),
                'timestamp': timestamp,
                'absolute_path': os.path.abspath(path)
            }
            
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    return manifest

def verify_manifest(manifest_path: str, logger=None) -> Dict:
    """
    Verify current files against stored manifest.
    For each file: check path exists, row count matches, MD5 matches.
    Returns dict: {filepath: {'status': 'ok'|'missing'|'row_mismatch'|'md5_mismatch', ...}}
    """
    if not os.path.exists(manifest_path):
        if logger:
            logger.warning(f"Manifest not found at {manifest_path}")
        return {}
        
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    results = {}
    base_dir = os.path.dirname(manifest_path)
    
    for rel_path, expected in manifest.items():
        abs_path = os.path.join(base_dir, rel_path)
        
        if not os.path.exists(abs_path):
            results[rel_path] = {'status': 'missing'}
            if logger:
                logger.error(f"Integrity check failed: File missing {rel_path}")
            continue
            
        current_rows = _count_rows(abs_path)
        if current_rows != expected['row_count']:
            results[rel_path] = {
                'status': 'row_mismatch',
                'expected_rows': expected['row_count'],
                'actual_rows': current_rows
            }
            if logger:
                logger.error(f"Integrity check failed: Row mismatch in {rel_path} (expected {expected['row_count']}, got {current_rows})")
            continue
            
        current_md5 = _calculate_md5(abs_path)
        if current_md5 != expected['md5_hash']:
            results[rel_path] = {
                'status': 'md5_mismatch',
                'expected_md5': expected['md5_hash'],
                'actual_md5': current_md5
            }
            if logger:
                logger.error(f"Integrity check failed: MD5 mismatch in {rel_path}")
            continue
            
        results[rel_path] = {'status': 'ok'}
        
    return results

def run_integrity_suite(data_dir: str, manifest_path: str, logger=None, quarantine_dir: Optional[str] = None) -> Dict:
    """
    Run the full integrity suite:
    1. Check all JSONL files for corruption, quarantine bad lines
    2. Deduplicate
    3. Verify manifest
    4. If manifest mismatches, log and return list of files needing re-download
    Returns: dict summary with all counts
    """
    summary = {
        'corrupt_files': 0,
        'total_quarantined': 0,
        'duplicate_files': 0,
        'total_duplicates_removed': 0,
        'manifest_failures': [],
        'manifest_ok': True
    }
    
    if not os.path.exists(data_dir):
        return summary
        
    # Find all JSONL files
    jsonl_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
                
    # 1 & 2. Corruption & Deduplication
    for filepath in jsonl_files:
        quarantine_path = os.path.join(quarantine_dir, os.path.basename(filepath) + ".quarantine") if quarantine_dir else None
        
        # Check corruption
        corr_res = check_jsonl_corruption(filepath, quarantine_path)
        if corr_res['corrupt_count'] > 0:
            summary['corrupt_files'] += 1
            summary['total_quarantined'] += corr_res['quarantined_count']
            if logger:
                logger.warning(f"Integrity: Found {corr_res['corrupt_count']} corrupt lines in {filepath}")
                
        # Check duplicates
        dedup_res = deduplicate_jsonl(filepath)
        if dedup_res['duplicate_count'] > 0:
            summary['duplicate_files'] += 1
            summary['total_duplicates_removed'] += dedup_res['duplicate_count']
            if logger:
                logger.warning(f"Integrity: Removed {dedup_res['duplicate_count']} duplicates from {filepath}")
                
    # 3. Verify manifest
    if os.path.exists(manifest_path):
        manifest_res = verify_manifest(manifest_path, logger)
        for rel_path, status_info in manifest_res.items():
            if status_info['status'] != 'ok':
                summary['manifest_failures'].append(rel_path)
                summary['manifest_ok'] = False
                
    return summary

def check_eval_duplicates(eval_dir: str, logger=None) -> Dict:
    """
    Check evaluation CSV files for duplicate pairs (same stereo+anti sentence pair).
    Returns: dict {file: duplicate_count}
    """
    results = {}
    if not os.path.exists(eval_dir):
        return results
        
    for root, _, files in os.walk(eval_dir):
        for file in files:
            if not file.endswith('.csv'):
                continue
                
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    # Determine columns
                    if not reader.fieldnames:
                        continue
                    
                    sent1_col = None
                    sent2_col = None
                    
                    for col in ['sent_more', 'stereo', 'stereotypical']:
                        if col in reader.fieldnames:
                            sent1_col = col
                            break
                    for col in ['sent_less', 'anti', 'anti-stereotypical']:
                        if col in reader.fieldnames:
                            sent2_col = col
                            break
                            
                    if not sent1_col or not sent2_col:
                        continue
                        
                    seen_pairs = set()
                    dup_count = 0
                    
                    for row in reader:
                        if sent1_col in row and sent2_col in row:
                            pair = (row[sent1_col].strip().lower(), row[sent2_col].strip().lower())
                            if pair in seen_pairs:
                                dup_count += 1
                            else:
                                seen_pairs.add(pair)
                                
                    if dup_count > 0:
                        results[filepath] = dup_count
                        if logger:
                            logger.warning(f"Integrity: Found {dup_count} duplicate pairs in evaluation set {filepath}")
                            
            except Exception as e:
                if logger:
                    logger.error(f"Error checking eval duplicates for {filepath}: {e}")
                    
    return results
