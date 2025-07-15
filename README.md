# Chinese-English Translation with Transformer

A PyTorch implementation of the Transformer model for Chinese-to-English translation, based on the "Attention Is All You Need" paper.

## Project Structure

```
translater/
├── config.py                 # Configuration settings
├── train.py                  # Training script
├── inference.py              # Translation inference
├── requirements.txt          # Dependencies
├── src/
│   ├── model/
│   │   └── transformer.py    # Transformer model implementation
│   ├── data/
│   │   └── dataset.py        # Data loading and preprocessing
│   └── utils/
│       └── tokenizer.py      # SentencePiece tokenizer utilities
├── scripts/
│   ├── download_data.py      # Download dataset
│   └── evaluate.py           # Model evaluation
├── tokenizer/                # Trained tokenizer models
├── model/                    # Trained model checkpoints
└── logs/                     # TensorBoard logs
```

## Features

- **Complete Transformer Implementation**: Multi-head attention, positional encoding, encoder-decoder architecture
- **SentencePiece Tokenization**: BPE tokenization for both Chinese and English
- **MPS Support**: Optimized for Apple Silicon (M1/M2) devices
- **Training Features**: Label smoothing, learning rate scheduling, gradient clipping
- **Inference Options**: Greedy decoding and beam search
- **Evaluation**: BLEU score calculation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset:
```bash
python scripts/download_data.py
```

## Training

Start training with the default configuration:
```bash
python train.py
```

The training will:
- Automatically train SentencePiece tokenizers if not present
- Download the YogeLiu/zh-en-translation-dataset from Hugging Face
- Train for 10 epochs with the specified parameters
- Save checkpoints and the best model

### Training Parameters
- Batch size: 32
- Learning rate: 0.0001 (with warmup)
- Device: MPS (Apple Silicon optimized)
- Model size: 6 layers, 8 heads, 512 dimensions

## Inference

Translate Chinese text to English:
```bash
python inference.py --text "你好世界" --beam_size 1
```

Options:
- `--text`: Chinese text to translate
- `--beam_size`: Beam size for beam search (1 for greedy decoding)
- `--max_length`: Maximum length of translation
- `--model_path`: Path to trained model

## Evaluation

Evaluate the model on test data:
```bash
python scripts/evaluate.py
```

This will calculate BLEU scores on the validation set.

## Model Architecture

The Transformer model follows the original paper:
- **Encoder**: 6 layers with multi-head self-attention
- **Decoder**: 6 layers with masked self-attention and encoder-decoder attention
- **Attention**: 8 heads with 64-dimensional keys/values
- **Feed-forward**: 2048 hidden dimensions
- **Vocabulary**: 32,000 tokens for both source and target

## Dataset

Uses the YogeLiu/zh-en-translation-dataset containing 1M Chinese-English sentence pairs.

Format:
```
src: 表演结束后，众人期待已久的园游会终于正式开锣，美味可口的素食佳肴让大家一饱口福。
tgt: after the performances, a garden party featuring delicious vegetarian food, which had been long awaited by many, finally began.
```

## Configuration

Modify `config.py` to adjust:
- Model hyperparameters
- Training settings
- Paths and device settings
- Dataset configuration