#!/usr/bin/env python3
"""
Russian Dataset Converter for MUS-CUDA
 Converts Russian text datasets to binary token format

Usage:
    python convert_russian_data.py --output data/russian_train \
        --files data/premise_question_answer4.txt \
                 data/premise_question_answer5.txt \
                 data/names_f.txt
"""

import struct
import sys
import os
import argparse
from pathlib import Path
import re

# Token IDs based on Uragan 1.0 vocabulary
PAD_TOKEN = 2001  # <PAD>
UNK_TOKEN = 2002  # <UNK>
BOS_TOKEN = 2003  # <BOS>
EOS_TOKEN = 2004  # <EOS>

class SimpleCharTokenizer:
    def __init__(self):
        self.next_id = 5000
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Add special tokens
        self._add_special()
        
    def _add_special(self):
        for i, name in enumerate(['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<NL>', '<T>', '<Q>', '<A>']):
            tid = 2001 + i
            self.char_to_id[name] = tid
            self.id_to_char[tid] = name
            
        # Add digits 0-9
        for i in range(10):
            self.char_to_id[str(i)] = 3000 + i
            self.id_to_char[3000 + i] = str(i)
            
    def encode(self, text: str) -> list:
        tokens = [BOS_TOKEN]
        for char in text:
            if char not in self.char_to_id:
                # Assign new ID
                self.char_to_id[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1
            tokens.append(self.char_to_id[char])
        tokens.append(EOS_TOKEN)
        return tokens
    
    def decode(self, tokens: list) -> str:
        result = []
        for t in tokens:
            if t in self.id_to_char:
                result.append(self.id_to_char[t])
            elif t == BOS_TOKEN:
                result.append('<BOS>')
            elif t == EOS_TOKEN:
                result.append('<EOS>')
            elif t == PAD_TOKEN:
                break
            else:
                result.append(f'<UNK:{t}>')
        return ''.join(result)

def parse_qa_file(path: str) -> list:
    """Parse premise_question_answer*.txt format"""
    samples = []
    with open(path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig strips BOM
        content = f.read()
    
    # Split by double newlines (empty lines between samples)
    blocks = content.split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        premise = ""
        question = ""
        answer = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('T:'):
                premise = line[2:].strip()
            elif line.startswith('Q:'):
                question = line[2:].strip()
            elif line.startswith('A:'):
                answer = line[2:].strip()
        
        if premise and question and answer:
            # Format: T: premise Q: question A: answer
            full_text = f"T: {premise} Q: {question} A: {answer}"
            samples.append(full_text)
    
    return samples

def parse_names_file(path: str) -> list:
    """Parse names_f.txt format (one name per line)"""
    samples = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            name = line.strip()
            if name:
                samples.append(name)
    return samples

def create_samples_from_file(path: str, file_type: str) -> list:
    """Load samples from a file based on its type"""
    if 'premise' in path.lower() or 'question' in path.lower():
        return parse_qa_file(path)
    elif 'name' in path.lower():
        return parse_names_file(path)
    else:
        # Generic text file - one line = one sample
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

def tokenize_samples(samples: list, tokenizer: SimpleCharTokenizer, max_len: int = 512) -> tuple:
    """
    Convert text samples to token IDs
    Returns: (tokens_list, labels_list)
    """
    all_tokens = []
    all_labels = []
    
    for sample in samples:
        tokens = tokenizer.encode(sample)
        
        # Pad to max_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        # Labels are the same as input, shifted by 1 (next token prediction)
        # For simplicity, we'll use the same tokens as labels
        labels = tokens.copy()
        
        # Pad if needed
        while len(tokens) < max_len:
            tokens.append(PAD_TOKEN)
            labels.append(-100)  # Ignore in loss
        
        all_tokens.extend(tokens)
        all_labels.extend(labels)
    
    return all_tokens, all_labels

def save_binary(tokens: list, labels: list, output_prefix: str):
    """Save as binary files for MUS-CUDA"""
    # Save tokens
    token_path = f"{output_prefix}_cache.bin"
    with open(token_path, 'wb') as f:
        f.write(struct.pack(f'{len(tokens)}i', *tokens))
    print(f"  Saved tokens: {token_path} ({len(tokens)} tokens)")
    
    # Save labels  
    label_path = f"{output_prefix}_labels.bin"
    with open(label_path, 'wb') as f:
        f.write(struct.pack(f'{len(labels)}q', *labels))
    print(f"  Saved labels: {label_path} ({len(labels)} labels)")

def main():
    parser = argparse.ArgumentParser(description='Convert Russian text to MUS tokens')
    parser.add_argument('--output', '-o', required=True, help='Output prefix (e.g., data/russian_train)')
    parser.add_argument('--files', '-f', nargs='+', required=True, help='Input text files')
    parser.add_argument('--max-len', type=int, default=512, help='Max sequence length')
    
    args = parser.parse_args()
    
    print("=== Russian Dataset Converter for MUS-CUDA ===\n")
    
    # Initialize tokenizer
    tokenizer = SimpleCharTokenizer()
    print(f"Tokenizer: SimpleChar with {len(tokenizer.char_to_id)} initial chars")
    
    # Process each file
    all_samples = []
    for filepath in args.files:
        print(f"\nProcessing: {filepath}")
        samples = create_samples_from_file(filepath, filepath)
        print(f"  Loaded {len(samples)} samples")
        
        # Show sample safely
        if samples:
            sample = samples[0][:50].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            print(f"  Sample: {sample}...")
        
        all_samples.extend(samples)
    
    print(f"\nTotal samples: {len(all_samples)}")
    # Show sample safely
    if all_samples:
        sample_preview = all_samples[0][:50].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        print(f"  First sample: {sample_preview}...")
    
    # Tokenize
    print("\nTokenizing...")
    tokens, labels = tokenize_samples(all_samples, tokenizer, args.max_len)
    
    # Save
    print(f"\nSaving to {args.output}...")
    save_binary(tokens, labels, args.output)
    
    # Stats
    num_seqs = len(tokens) // args.max_len
    print(f"\n=== Summary ===")
    print(f"  Sequences: {num_seqs}")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Vocab size: {tokenizer.next_id}")
    
    # Print vocab stats
    print(f"\n=== Token Type Distribution ===")
    aer = sum(1 for t in tokens if t < 2001)
    text = sum(1 for t in tokens if t >= 2000)
    print(f"  AER tokens: {aer} ({100*aer/len(tokens):.1f}%)")
    print(f"  TEXT tokens: {text} ({100*text/len(tokens):.1f}%)")

if __name__ == '__main__':
    main()