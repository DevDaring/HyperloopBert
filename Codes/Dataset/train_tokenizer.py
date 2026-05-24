import os
import sys
import json
import logging
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

# Add parent dir to path to import common
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logging_setup import setup_logging

logger = setup_logging('train_tokenizer')

def train_wordpiece_tokenizer(train_file_path: str, output_dir: str, vocab_size: int = 30522):
    """
    Train a WordPiece tokenizer from scratch on the training corpus.
    Standard BERT special tokens: [PAD], [UNK], [CLS], [SEP], [MASK].
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(train_file_path):
        logger.error(f"Training corpus not found at {train_file_path}")
        return
        
    logger.info(f"Training WordPiece tokenizer (vocab size: {vocab_size}) on {train_file_path}...")
    
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    
    # Standard BERT normalization (lowercase, remove accents, handle unicode)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    
    # Standard BERT pre-tokenization (split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    
    # Set up the trainer
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2
    )
    
    # Generator to stream lines from the JSONL file without loading it all into memory
    def batch_iterator():
        with open(train_file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "text" in obj:
                        yield obj["text"]
                except json.JSONDecodeError:
                    continue

    # Train the tokenizer
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Add post-processor to automatically add [CLS] and [SEP]
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    
    # Add decoder
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    
    # Wrap in HF PreTrainedTokenizerFast for easy use in pipelines
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    
    # Save the tokenizer
    fast_tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer trained and saved to {output_dir}")

if __name__ == "__main__":
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    TRAIN_PATH = os.path.join(DATA_DIR, 'fineweb-edu', 'train_raw.jsonl')
    OUT_DIR = os.path.join(DATA_DIR, 'tokenizer')
    
    train_wordpiece_tokenizer(TRAIN_PATH, OUT_DIR)
