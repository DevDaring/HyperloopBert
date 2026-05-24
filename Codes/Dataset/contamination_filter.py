import os
import sys
import json
import logging
from typing import Set, List, Dict
from tqdm import tqdm
import pandas as pd

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.llm_utils import call_llm
from common.env_loader import env_loader

logger = setup_logging('contamination_filter')

def extract_ngrams(text: str, n: int = 8) -> Set[str]:
    """Extract character or word n-grams. Using word n-grams here."""
    words = text.lower().split()
    if len(words) < n:
        return set([" ".join(words)])
    
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngrams.add(" ".join(words[i:i+n]))
    return ngrams

def load_eval_ngrams(eval_dir: str, n: int = 8) -> Set[str]:
    """
    Load all evaluation datasets and extract n-grams to build a contamination filter.
    Includes CSVs for bias datasets and GLUE if available.
    """
    eval_ngrams = set()
    
    if not os.path.exists(eval_dir):
        logger.warning(f"Eval dir {eval_dir} not found. Returning empty ngram set.")
        return eval_ngrams
        
    for root, _, files in os.walk(eval_dir):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                try:
                    df = pd.read_csv(path)
                    for col in df.columns:
                        # Only process columns that likely contain text
                        if df[col].dtype == object:
                            for text in df[col].dropna():
                                if isinstance(text, str):
                                    eval_ngrams.update(extract_ngrams(text, n))
                except Exception as e:
                    logger.error(f"Error processing {path} for ngrams: {e}")
                    
    logger.info(f"Extracted {len(eval_ngrams)} unique {n}-grams from evaluation datasets.")
    return eval_ngrams

def filter_training_data(input_path: str, output_path: str, eval_ngrams: Set[str], n: int = 8):
    """
    Stream training data and write clean lines to output_path.
    If a line contains any eval n-gram, it is flagged and discarded.
    Saves a few flagged examples for LLM review.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input path {input_path} not found.")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_docs = 0
    clean_docs = 0
    flagged_docs = 0
    
    flagged_examples = []
    
    logger.info(f"Filtering {input_path} to {output_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
         
        for line in tqdm(f_in, desc="Filtering Contamination"):
            total_docs += 1
            try:
                obj = json.loads(line)
                text = obj.get('text', '')
                
                doc_ngrams = extract_ngrams(text, n)
                overlap = doc_ngrams.intersection(eval_ngrams)
                
                if overlap:
                    flagged_docs += 1
                    if len(flagged_examples) < 10:
                        flagged_examples.append({
                            'text': text[:500] + "...",  # truncate for review
                            'overlap': list(overlap)[:3]
                        })
                else:
                    f_out.write(line)
                    clean_docs += 1
                    
            except json.JSONDecodeError:
                continue
                
    logger.info(f"Filtering complete. Total: {total_docs}, Clean: {clean_docs}, Flagged: {flagged_docs}")
    return flagged_examples

def llm_sanity_check(flagged_examples: List[Dict]):
    """
    Send a sample of flagged examples to the primary LLM to ask if it looks like
    genuine contamination or a false positive (e.g., generic boilerplate).
    """
    if not flagged_examples:
        logger.info("No flagged examples to review.")
        return
        
    if not env_loader.is_provider_available('gemini'):
        logger.info("Gemini provider not available. Skipping LLM sanity check.")
        return
        
    logger.info("Running LLM sanity check on flagged examples...")
    
    prompt = """
    We are filtering a pretraining dataset to prevent evaluation leakage. 
    We flagged the following documents because they contain 8-gram overlaps with our evaluation sets 
    (bias benchmarks like CrowS-Pairs, WinoBias, and GLUE tasks).
    
    Please review these examples and determine if the overlap is likely genuine contamination 
    (the exact test question/sentence is in the text) or a false positive (common boilerplate phrase).
    
    Examples:
    {examples}
    
    Return a JSON object with a single key "analysis" containing a brief summary.
    """
    
    examples_str = ""
    for i, ex in enumerate(flagged_examples[:5]):
        examples_str += f"Example {i+1}:\nOverlap N-grams: {ex['overlap']}\nText Snippet: {ex['text']}\n\n"
        
    schema = {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"}
        },
        "required": ["analysis"]
    }
    
    try:
        response = call_llm(
            prompt.format(examples=examples_str),
            provider='gemini',
            schema=schema,
            temperature=0.1
        )
        logger.info("LLM Contamination Sanity Check Result:")
        logger.info(response.get('analysis', str(response)))
    except Exception as e:
        logger.error(f"LLM sanity check failed: {e}")

if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    EVAL_DIR = os.path.join(DATA_DIR, 'datasets_eval')
    TRAIN_RAW = os.path.join(DATA_DIR, 'fineweb-edu', 'train_raw.jsonl')
    TRAIN_FILTERED = os.path.join(DATA_DIR, 'fineweb-edu', 'train_filtered.jsonl')
    
    ngrams = load_eval_ngrams(EVAL_DIR, n=8)
    
    if ngrams:
        flagged = filter_training_data(TRAIN_RAW, TRAIN_FILTERED, ngrams, n=8)
        if flagged:
            llm_sanity_check(flagged)
    else:
        logger.warning("No evaluation n-grams loaded. Simply copying raw to filtered.")
        import shutil
        if os.path.exists(TRAIN_RAW):
            shutil.copy2(TRAIN_RAW, TRAIN_FILTERED)
