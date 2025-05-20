import os
import numpy as np
import torch
import pandas as pd
from time import time
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import gc
import csv
model_name = "model name"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,       
    device_map="auto"                
)

model.eval()                          
model.gradient_checkpointing_enable()  

print("Model loaded Successfully!!!")

def convert_labels_to_integers(labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return labels, label_encoder

def calculate_intrinsic_metrics(perword_features, original_words):
    total_tokens = sum(len(subword) for sentence in perword_features for subword in sentence)
    vocab = set(token for sentence in perword_features for subword in sentence for token in subword)
    continued_words = sum(1 for sentence in perword_features for subword in sentence if len(subword) > 1)
    total_words = sum(len(sentence) for sentence in perword_features)
    return {
        'fertility': total_tokens / total_words if total_words > 0 else 0,
        'TTR': len(vocab) / total_tokens if total_tokens > 0 else 0,
        'proportion_continued_words': continued_words / total_words if total_words > 0 else 0
    }

def calculate_extrinsic_metrics(model, features, pad_token_id, fertility, batch_size=4, log_csv_path='ctoken_log_fix2.csv'):
    model_device = next(model.parameters()).device
    config = model.config
    num_layers = getattr(config, "num_hidden_layers", 32)
    hidden_size = getattr(config, "hidden_size", 2560)
    vocab_size = getattr(config, "vocab_size", 50000)

    total_C = 0
    total_tokens = 0
    total_batches = 0

    start_time = time()

    with open(log_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'batch_idx', 'example_idx', 'real_token_count', 'C_i'
        ])

        for i in range(0, len(features), batch_size):
            batch = features[i:i + batch_size]
            batch_tensors = [torch.tensor(seq, dtype=torch.long).to(model_device) for seq in batch]
            inputs = pad_sequence(batch_tensors, batch_first=True, padding_value=pad_token_id).to(model_device)
            attention_mask = (inputs != pad_token_id).long().to(model_device)

            try:
                with torch.no_grad():
                    _ = model(input_ids=inputs, attention_mask=attention_mask)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"Skipping batch due to CUDA error: {e}")
                    continue
                else:
                    raise e

            real_lengths = attention_mask.sum(dim=1)

            for j in range(len(real_lengths)):
                L = real_lengths[j].item()
                C_i = 96 * num_layers * hidden_size**2 * (
                    1 + (L / (6 * hidden_size)) + (vocab_size / (16 * num_layers * hidden_size))
                )
                total_C += C_i
                total_tokens += L

                writer.writerow([
                    total_batches + 1, j, L, C_i
                ])

            total_batches += 1

            torch.cuda.empty_cache()
            gc.collect()

    inference_time = time() - start_time

    if total_tokens == 0:
        raise ValueError("No valid batches processed. total_tokens is 0.")

    final_C_token = total_C / total_tokens
    final_C_word = final_C_token * fertility

    return {
        'inference_speed': inference_time,
        'energy_consumption': inference_time * 0.05,
        'computational_cost_per_token': final_C_token,
        'computational_cost_per_word': final_C_word
    }

base_path = 'Path'
datasets = ['Tense', 'Number_Noun', 'Number_Pronoun']
splits = ['train', 'test', 'dev']
csv_file_path = "Path"
results = []

for dataset in datasets:
    print(f" Processing dataset: {dataset}")
    for split in splits:
        print(f"Running split: {split}")
        base = os.path.join(base_path, dataset)
        sentence_file = os.path.join(base, f"{split}.csv_sentence_features.npy")
        perword_file = os.path.join(base, f"{split}.csv_perword_features.npy")
        words_file = os.path.join(base, f"{split}.csv_original_words.npy")
        labels_file = os.path.join(base, f"{split}.csv_labels.npy")

        if all(os.path.exists(f) for f in [sentence_file, perword_file, words_file, labels_file]):
            try:
                sentence_features = np.load(sentence_file, allow_pickle=True).tolist()
                perword_features = np.load(perword_file, allow_pickle=True).tolist()
                original_words = np.load(words_file, allow_pickle=True).tolist()
                labels = np.load(labels_file, allow_pickle=True)

                if len(sentence_features) == 0 or len(labels) == 0:
                    print(f"Skipping empty split: {split}")
                    continue

                labels, label_encoder = convert_labels_to_integers(labels)
                intrinsic_metrics = calculate_intrinsic_metrics(perword_features, original_words)
                pad_token_id = max(max(seq) for seq in sentence_features if seq) + 1

                log_path = f'{dataset}_{split}_ctoken_log.csv'
                extrinsic_metrics = calculate_extrinsic_metrics(
                    model, sentence_features, pad_token_id,
                    fertility=intrinsic_metrics['fertility'],
                    log_csv_path=log_path
                )

                results.append({
                    "dataset": dataset,
                    "split": split,
                    "samples_processed": len(sentence_features),
                    **intrinsic_metrics,
                    **extrinsic_metrics
                })

            except Exception as e:
                print(f"Error processing {dataset} {split}: {e}")
        else:
            print(f"Missing files for {dataset}/{split}")

df_results = pd.DataFrame(results)
df_results.to_csv(csv_file_path, index=False)
print(f"\nResults saved to {csv_file_path}")

for _, row in df_results.iterrows():
    print(f"\nDataset: {row['dataset']} | Split: {row['split']}")
    print(f"  Total samples: {row['samples_processed']}")
    print("  Intrinsic:")
    for metric in ["fertility", "TTR", "proportion_continued_words"]:
        print(f"    {metric}: {row[metric]}")
    print("  Extrinsic:")
    for metric in ["inference_speed", "energy_consumption", "computational_cost_per_token", "computational_cost_per_word"]:
        print(f"    {metric}: {row[metric]}")
