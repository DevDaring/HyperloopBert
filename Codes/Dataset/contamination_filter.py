import os
import sys
import json
import logging
from typing import Set, List, Dict
from tqdm import tqdm
import pandas as pd

# Aho-Corasick for O(document length) multi-substring matching. Without it the
# short-phrase check is O(docs x phrases) -- ~13 docs/s on 36k phrases, i.e.
# ~20h for a 1M-doc corpus. With it, the whole corpus filters in minutes.
try:
    import ahocorasick
    _AHOCORASICK_AVAILABLE = True
except ImportError:
    _AHOCORASICK_AVAILABLE = False

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging
from common.llm_utils import call_judge
from common.env_loader import env_loader

logger = setup_logging('contamination_filter')

def extract_ngrams(text: str, n: int = 8) -> Set[str]:
    """Extract word n-grams (lowercased)."""
    words = text.lower().split()
    if len(words) < n:
        return set()
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngrams.add(" ".join(words[i:i+n]))
    return ngrams


def extract_short_phrase(text: str, n: int = 8, min_words: int = 4):
    """
    Eval sentences SHORTER than the n-gram window can never intersect a
    training document's n-gram set, silently exempting them from
    decontamination. Return the whole lowercased phrase for substring
    matching instead (phrases under min_words are too generic to match on).
    """
    words = text.lower().split()
    if min_words <= len(words) < n:
        return " ".join(words)
    return None

def load_eval_ngrams(eval_dir: str, n: int = 8):
    """
    Build the contamination filter from ALL evaluation text: bias-dataset CSVs
    on disk PLUS the GLUE task sentences loaded from HF (GLUE is evaluated
    from HF at fine-tuning time, so it must be screened here too).

    Returns (eval_ngrams, short_phrases): word n-grams for standard matching,
    and whole short phrases (< n words) for substring matching.
    """
    eval_ngrams: Set[str] = set()
    short_phrases: Set[str] = set()

    def add_text(text):
        if isinstance(text, str) and text:
            eval_ngrams.update(extract_ngrams(text, n))
            phrase = extract_short_phrase(text, n)
            if phrase:
                short_phrases.add(phrase)

    if os.path.exists(eval_dir):
        for root, _, files in os.walk(eval_dir):
            for file in files:
                if file.endswith('.csv'):
                    path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(path)
                        for col in df.columns:
                            if df[col].dtype == object:
                                for text in df[col].dropna():
                                    add_text(text)
                    except Exception as e:
                        logger.error(f"Error processing {path} for ngrams: {e}")
    else:
        logger.warning(f"Eval dir {eval_dir} not found.")

    # GLUE sentences (SST-2, RTE, MRPC, QNLI - the tasks this project evaluates)
    try:
        from datasets import load_dataset
        glue_keys = {'sst2': ['sentence'], 'rte': ['sentence1', 'sentence2'],
                     'mrpc': ['sentence1', 'sentence2'], 'qnli': ['question', 'sentence']}
        for task, keys in glue_keys.items():
            try:
                for split in ('train', 'validation'):
                    ds = load_dataset('glue', task, split=split)
                    for ex in ds:
                        for key in keys:
                            add_text(ex.get(key))
                logger.info(f"Screened GLUE/{task} into the contamination filter.")
            except Exception as e:
                logger.warning(f"Could not screen GLUE/{task}: {e} - GLUE results "
                               f"for this task may carry contamination risk.")
    except Exception as e:
        logger.warning(f"GLUE screening unavailable ({e}).")

    logger.info(f"Extracted {len(eval_ngrams)} unique {n}-grams and "
                f"{len(short_phrases)} short phrases from evaluation datasets.")
    return eval_ngrams, short_phrases

def filter_training_data(input_path: str, output_path: str, eval_ngrams: Set[str],
                         n: int = 8, short_phrases: Set[str] = None):
    """
    Stream training data and write clean lines to output_path.
    A document is flagged and discarded if it shares any n-gram with the
    evaluation sets OR contains any short eval phrase as a substring.
    Saves a few flagged examples for LLM review.
    """
    short_phrases = short_phrases or set()
    if not os.path.exists(input_path):
        logger.error(f"Input path {input_path} not found.")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_docs = 0
    clean_docs = 0
    flagged_docs = 0
    
    flagged_examples = []
    
    logger.info(f"Filtering {input_path} to {output_path}...")

    # Build an Aho-Corasick automaton over the short phrases once (O(doc length)
    # matching per document instead of O(phrases)). Falls back to the slow
    # substring scan only if pyahocorasick is unavailable (logged loudly).
    automaton = None
    if short_phrases and _AHOCORASICK_AVAILABLE:
        automaton = ahocorasick.Automaton()
        for phrase in short_phrases:
            if phrase:
                automaton.add_word(phrase, phrase)
        automaton.make_automaton()
        logger.info(f"Aho-Corasick automaton built over {len(short_phrases)} short phrases.")
    elif short_phrases and not _AHOCORASICK_AVAILABLE:
        logger.warning("pyahocorasick not installed; short-phrase matching will be "
                       "O(docs x phrases) and MUCH slower. `pip install pyahocorasick`.")

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, desc="Filtering Contamination"):
            total_docs += 1
            try:
                obj = json.loads(line)
                text = obj.get('text', '')

                doc_ngrams = extract_ngrams(text, n)
                overlap = doc_ngrams.intersection(eval_ngrams)

                if not overlap and short_phrases:
                    text_lower = text.lower()
                    if automaton is not None:
                        hits = []
                        for _, phrase in automaton.iter(text_lower):
                            hits.append(phrase)
                            if len(hits) >= 3:
                                break
                    else:
                        hits = [p for p in short_phrases if p in text_lower]
                    if hits:
                        overlap = set(hits[:3])

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
        
    # Judgement policy: DeepSeek -> Mistral -> OpenRouter (never Gemini).
    if not any(env_loader.is_provider_available(p)
               for p in ('deepseek', 'mistral', 'openrouter')):
        logger.info("No judge provider (DeepSeek/Mistral/OpenRouter) available. "
                    "Skipping LLM sanity check.")
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
        response = call_judge(
            prompt.format(examples=examples_str),
            schema=schema,
            temperature=0.1
        )
        logger.info("LLM Contamination Sanity Check Result:")
        logger.info(response.get('analysis', str(response)))
    except Exception as e:
        logger.error(f"LLM sanity check failed: {e}")

if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    EVAL_DIR = os.path.join(DATA_DIR, 'datasets_eval')
    TRAIN_RAW = os.path.join(DATA_DIR, 'fineweb-edu', 'train_raw.jsonl')
    TRAIN_FILTERED = os.path.join(DATA_DIR, 'fineweb-edu', 'train_filtered.jsonl')
    
    ngrams, short_phrases = load_eval_ngrams(EVAL_DIR, n=8)

    if ngrams or short_phrases:
        flagged = filter_training_data(TRAIN_RAW, TRAIN_FILTERED, ngrams, n=8,
                                       short_phrases=short_phrases)
        if flagged:
            llm_sanity_check(flagged)
    else:
        logger.warning("No evaluation n-grams loaded. Simply copying raw to filtered.")
        import shutil
        if os.path.exists(TRAIN_RAW):
            shutil.copy2(TRAIN_RAW, TRAIN_FILTERED)
