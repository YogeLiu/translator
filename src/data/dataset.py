import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm
from config import Config


class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer, max_seq_len=512):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item["src"]
        tgt_text = item["tgt"]

        src_tokens = self.src_tokenizer.encode(src_text, out_type=int)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text, out_type=int)

        src_tokens = [self.src_tokenizer.bos_id()] + src_tokens + [self.src_tokenizer.eos_id()]
        tgt_tokens = [self.tgt_tokenizer.bos_id()] + tgt_tokens + [self.tgt_tokenizer.eos_id()]

        src_tokens = src_tokens[: self.max_seq_len]
        tgt_tokens = tgt_tokens[: self.max_seq_len]

        return {"src": torch.tensor(src_tokens, dtype=torch.long), "tgt": torch.tensor(tgt_tokens, dtype=torch.long), "src_len": len(src_tokens), "tgt_len": len(tgt_tokens)}


def collate_fn(batch):
    src_max_len = max([item["src_len"] for item in batch])
    tgt_max_len = max([item["tgt_len"] for item in batch])

    src_batch = []
    tgt_input_batch = []
    tgt_output_batch = []

    for item in batch:
        src = item["src"]
        tgt = item["tgt"]

        src_padded = torch.cat([src, torch.full((src_max_len - len(src),), 0, dtype=torch.long)])

        tgt_padded = torch.cat([tgt, torch.full((tgt_max_len - len(tgt),), 0, dtype=torch.long)])

        src_batch.append(src_padded)
        tgt_input_batch.append(tgt_padded[:-1])
        tgt_output_batch.append(tgt_padded[1:])

    return {"src": torch.stack(src_batch), "tgt_input": torch.stack(tgt_input_batch), "tgt_output": torch.stack(tgt_output_batch)}


def load_translation_data(config: Config):
    import random

    dataset = load_dataset(config.dataset_name)

    train_data = dataset["train"]
    val_data = dataset["validation"] if "validation" in dataset else dataset["train"].train_test_split(test_size=0.1)["test"]

    if config.is_demo:
        # 随机选择demo样本
        train_indices = random.sample(range(len(train_data)), min(config.demo_train_size, len(train_data)))
        val_indices = random.sample(range(len(val_data)), min(config.demo_val_size, len(val_data)))
        train_data = train_data.select(train_indices)
        val_data = val_data.select(val_indices)

    return train_data, val_data


def create_dataloaders(train_data, val_data, src_tokenizer, tgt_tokenizer, config: Config):
    train_dataset = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer, config.max_seq_len)
    val_dataset = TranslationDataset(val_data, src_tokenizer, tgt_tokenizer, config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=config.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=config.num_workers)

    return train_loader, val_loader
