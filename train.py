import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config import Config
from src.transformer.transformer import Transformer, create_padding_mask, create_look_ahead_mask
from src.data.dataset import load_translation_data, create_dataloaders
from src.utils.tokenizer import train_tokenizer, load_tokenizers


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.05, ignore_index=0):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)

        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * torch.log_softmax(pred, dim=1), dim=1))


def create_masks(src, tgt, pad_idx):
    src_mask = create_padding_mask(src, pad_idx)
    tgt_mask = create_padding_mask(tgt, pad_idx)
    look_ahead_mask = create_look_ahead_mask(tgt.size(1)).to(tgt.device)
    tgt_mask = tgt_mask & look_ahead_mask
    return src_mask.to(src.device), tgt_mask.to(tgt.device)


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, config):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    smooth_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)

        src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx=0)

        output = model(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(output, tgt_output)

        loss = loss / config.gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.gradient_accumulation_steps
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss.item() if batch_idx > 0 else loss.item()

        current_lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({"smooth_loss": f"{smooth_loss:.4f}", "avg_loss": f"{total_loss / (batch_idx + 1):.4f}", "lr": f"{current_lr:.2e}"})

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")

        for batch in progress_bar:
            src = batch["src"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            tgt_output = batch["tgt_output"].to(device)

            src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx=0)

            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output, tgt_output)

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, epoch, loss, config):
    os.makedirs(config.model_path, exist_ok=True)

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "loss": loss,
        "config": config.__dict__,
    }

    checkpoint_path = os.path.join(config.model_path, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)

    best_model_path = os.path.join(config.model_path, "best_model.pt")
    torch.save(checkpoint, best_model_path)

    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    config = Config()
    print(f"Using device: {config.device}")

    if not os.path.exists(os.path.join(config.tokenizer_path, "src_tokenizer.model")):
        print("Training tokenizers...")
        train_tokenizer(config)

    print("Loading tokenizers...")
    src_tokenizer, tgt_tokenizer = load_tokenizers(config)

    print("Loading data...")
    train_data, val_data = load_translation_data(config)
    train_loader, val_loader = create_dataloaders(train_data, val_data, src_tokenizer, tgt_tokenizer, config)

    print("Initializing model...")
    model = Transformer(
        src_vocab_size=src_tokenizer.get_piece_size(), tgt_vocab_size=tgt_tokenizer.get_piece_size(), d_model=config.d_model, n_heads=config.n_heads, n_layers=config.n_layers, d_ff=config.d_ff, max_seq_len=config.max_seq_len, dropout=config.dropout
    ).to(config.device)

    if config.device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = LabelSmoothingLoss(vocab_size=tgt_tokenizer.get_piece_size(), smoothing=config.label_smoothing, ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

    writer = SummaryWriter(config.log_dir)

    resume_checkpoint = getattr(config, "resume_checkpoint", None)
    start_epoch = 0

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=config.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["loss"]
        print(f"Resuming training from epoch {start_epoch} with previous validation loss: {best_val_loss:.4f}")

    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, config.device, config)
        val_loss = validate(model, val_loader, criterion, config.device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.patience}")
            if patience_counter >= config.patience:
                print("Early stopping triggered!")
                break

    writer.close()
    print("Training completed!")


# push model and related files to hub
def push_model_to_hub(model, tokenizer, config):
    model.push_to_hub(config.model_name)
    tokenizer.push_to_hub(config.model_name)


if __name__ == "__main__":
    main()
