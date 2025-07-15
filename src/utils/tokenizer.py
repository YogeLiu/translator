import sentencepiece as spm
import os
from datasets import load_dataset
from config import Config


def train_tokenizer(config: Config):
    print("Loading dataset for tokenizer training...")
    dataset = load_dataset(config.dataset_name)
    train_data = dataset["train"]

    src_texts = []
    tgt_texts = []

    print("Extracting texts...")
    for item in train_data:
        src_texts.append(item["src"])
        tgt_texts.append(item["tgt"])

    os.makedirs(config.tokenizer_path, exist_ok=True)

    src_text_file = os.path.join(config.tokenizer_path, "src_texts.txt")
    tgt_text_file = os.path.join(config.tokenizer_path, "tgt_texts.txt")

    print("Writing text files...")
    with open(src_text_file, "w", encoding="utf-8") as f:
        for text in src_texts:
            f.write(text + "\n")

    with open(tgt_text_file, "w", encoding="utf-8") as f:
        for text in tgt_texts:
            f.write(text + "\n")

    print("Training source tokenizer...")
    spm.SentencePieceTrainer.train(
        input=src_text_file,
        model_prefix=os.path.join(config.tokenizer_path, "src_tokenizer"),
        vocab_size=config.vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=config.PAD_TOKEN,
        unk_piece=config.UNK_TOKEN,
        bos_piece=config.BOS_TOKEN,
        eos_piece=config.EOS_TOKEN,
        num_threads=config.num_workers,
    )

    print("Training target tokenizer...")
    spm.SentencePieceTrainer.train(
        input=tgt_text_file,
        model_prefix=os.path.join(config.tokenizer_path, "tgt_tokenizer"),
        vocab_size=config.vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=config.PAD_TOKEN,
        unk_piece=config.UNK_TOKEN,
        bos_piece=config.BOS_TOKEN,
        eos_piece=config.EOS_TOKEN,
        num_threads=config.num_workers,
    )

    print("Tokenizers trained successfully!")

    os.remove(src_text_file)
    os.remove(tgt_text_file)


def load_tokenizers(config: Config):
    src_tokenizer = spm.SentencePieceProcessor()
    tgt_tokenizer = spm.SentencePieceProcessor()

    src_tokenizer.load(os.path.join(config.tokenizer_path, "src_tokenizer.model"))
    tgt_tokenizer.load(os.path.join(config.tokenizer_path, "tgt_tokenizer.model"))

    return src_tokenizer, tgt_tokenizer


if __name__ == "__main__":
    config = Config()
    train_tokenizer(config)
