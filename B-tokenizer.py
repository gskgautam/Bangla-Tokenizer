import os
import pandas as pd
import numpy as np
import math
import json
import shutil
from collections import defaultdict, Counter

def compute_probabilities(sentences):
    unigram_counts = Counter()
    bigram_counts = Counter()
    total_unigrams = 0

    for sentence in sentences:
        words = sentence.strip().split()
        for word in words:
            for i in range(1, len(word)+1):
                unigram = word[:i]
                unigram_counts[unigram] += 1
                total_unigrams += 1
            for i in range(1, len(word)):
                bigram = (word[:i], word[i:])
                bigram_counts[bigram] += 1

    return unigram_counts, bigram_counts, total_unigrams

def compute_pmi(m1, m2, unigram_counts, bigram_counts, total_count):
    p_m1 = unigram_counts.get(m1, 1) / total_count
    p_m2 = unigram_counts.get(m2, 1) / total_count
    p_m1_m2 = bigram_counts.get((m1, m2), 1) / total_count
    return math.log2(p_m1_m2 / (p_m1 * p_m2) + 1e-9)

def bond_strength(m1, m2, unigram_counts, bigram_counts, total_count, alpha=1.0, beta=1.0):
    freq_score = math.log(unigram_counts.get(m1, 1))
    pmi_score = compute_pmi(m1, m2, unigram_counts, bigram_counts, total_count)
    len_score = math.log(len(m1) + 1)
    return freq_score + alpha * pmi_score + beta * len_score

def molecular_split(word, unigram_counts, bigram_counts, total_count,
                    alpha=1.0, beta=1.0):
    tokens = []
    i = 0
    while i < len(word):
        best_token = word[i]
        best_strength = float('-inf')
        best_j = i + 1
        for j in range(i+1, len(word)+1):
            morph = word[i:j]
            next_morph = word[j:j+4] if j+4 <= len(word) else word[j:]
            strength = bond_strength(morph, next_morph, unigram_counts, bigram_counts, total_count, alpha, beta)
            if strength > best_strength:
                best_token = morph
                best_j = j
                best_strength = strength
        tokens.append(best_token)
        i = best_j
    return tokens

def process_file(file_path, unigram_counts, bigram_counts, total_count):
    df = pd.read_csv(file_path)
    sentences = df['samples'].fillna('').tolist()
    labels = df['labels'].tolist()

    flat_tokens = []
    perword_tokens = []
    original_words = []

    for sentence in sentences:
        word_tokens = []
        sentence_flat = []
        words = sentence.split()
        for word in words:
            toks = molecular_split(word, unigram_counts, bigram_counts, total_count)
            sentence_flat.extend(toks)
            word_tokens.append(toks)
        flat_tokens.append(sentence_flat)
        perword_tokens.append(word_tokens)
        original_words.append(words)

    return flat_tokens, perword_tokens, original_words, labels

base_path = 'raw dataset path'
output_base = 'Output path'
TASK_FOLDERS = ['Tense', 'Number_Noun', 'Number_Pronoun']
SPLITS = ['train.csv', 'dev.csv', 'test.csv']

os.makedirs(output_base, exist_ok=True)

for task in TASK_FOLDERS:
    task_path = os.path.join(base_path, task)
    output_path = os.path.join(output_base, task)
    os.makedirs(output_path, exist_ok=True)

    # Load training sentences
    train_path = os.path.join(task_path, 'train.csv')
    df_train = pd.read_csv(train_path)
    all_sentences = df_train['samples'].dropna().tolist()

    # Compute frequency and bond info
    unigram_counts, bigram_counts, total_count = compute_probabilities(all_sentences)

    # Process splits
    for split in SPLITS:
        split_path = os.path.join(task_path, split)
        if not os.path.exists(split_path):
            continue

        flat_tokens, perword_tokens, original_words, labels = process_file(
            split_path, unigram_counts, bigram_counts, total_count)

        np.save(os.path.join(output_path, f'{split}_sentence_features.npy'), np.array(flat_tokens, dtype=object), allow_pickle=True)
        np.save(os.path.join(output_path, f'{split}_perword_features.npy'), np.array(perword_tokens, dtype=object), allow_pickle=True)
        np.save(os.path.join(output_path, f'{split}_original_words.npy'), np.array(original_words, dtype=object), allow_pickle=True)
        np.save(os.path.join(output_path, f'{split}_labels.npy'), np.array(labels), allow_pickle=True)

        print(f"Saved: {task}/{split}")

    # Zip task folder
    zip_file_path = f'{output_base}_{task}.zip'
    shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', output_path)
    print(f"Created zip archive: {zip_file_path}")