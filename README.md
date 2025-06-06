# Bangla Tokenization and Evaluation

This repository contains experiments and evaluations on various tokenization methods and language models for Bangla text classification and generation tasks.

## 📚 Datasets

We utilize the following datasets in our experiments:

- **BanglaTense**: Available at [Mendeley Data](https://data.mendeley.com/datasets/39w5khrg87/4)  
- **BanglaParaphrase**: Available at [GitHub - csebuetnlp/banglaparaphrase](https://github.com/csebuetnlp/banglaparaphrase)

## 🔤 Tokenization Methods

We explore several popular tokenization methods, listed below with their original sources:

- **BPE (Byte Pair Encoding)** — [Sennrich et al., 2016]([https://aclanthology.org/P16-1162](https://arxiv.org/abs/1508.07909))
- **Unigram Language Model** — [Kudo, 2018]([https://aclanthology.org/D18-2012](https://aclanthology.org/P18-1007/))
- **WordPiece** — [Wu et al., 2016](https://arxiv.org/abs/1609.08144)
- **Character-Level Tokenization** — [Ling et al., 2015]([https://aclanthology.org/P15-1162](https://aclanthology.org/D15-1176/))

We used the implementations provided by their respective authors or libraries. Readers are encouraged to refer to the original papers and official repositories for details.

## 🧠 Language Models Used

The following categories of models were used in our evaluation:

### Instruction-tuned Models
- [LLaMA3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
- [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)

### Multilingual Models
- [Bloom-1b1](https://huggingface.co/bigscience/bloom-1b1)
- [Bloomz-1b7](https://huggingface.co/bigscience/bloomz-1b7)

### Bangla-Specific Models
- [Bangla-s1k-qwen-2.5-3B-Instruct](https://huggingface.co/BanglaLLM/Bangla-s1k-qwen-2.5-3B-Instruct)

## 📊 Evaluation and Code

- **`Evaluation.py`**: Contains all scripts and metrics used for evaluating tokenizers and models.
- **`BanglaTokenizer.py`**: Implements our proposed Bangla-specific tokenizer.

🖥️ Note on Performance Variability:
Evaluation results may vary by up to ±10% depending on your hardware configuration, especially GPU type, memory bandwidth, and compute environment. This margin reflects differences in numerical precision, runtime optimizations, and stability of training dynamics during task vector extraction and calibration.
