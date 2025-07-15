import torch


class Config:
    # Model parameters
    d_model = 256
    n_heads = 8
    n_layers = 2
    d_ff = 512
    max_seq_len = 512
    dropout = 0.3

    # Training parameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 30
    warmup_steps = 500
    label_smoothing = 0.1

    # Learning rate scheduling
    use_cosine_scheduler = True
    cosine_restart_period = 10
    lr_min = 1e-6

    # Gradient accumulation
    gradient_accumulation_steps = 2

    # Early stopping
    patience = 5
    min_delta = 0.005

    # Data parameters
    dataset_name = "YogeLiu/zh-en-translation-dataset"
    src_lang = "zh"
    tgt_lang = "en"
    vocab_size = 32000

    # Paths
    model_path = "./model"
    tokenizer_path = "./tokenizer"
    log_dir = "./logs"

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    # Parallel
    num_workers = 8

    # demo
    is_demo = True
    demo_train_size = 100000
    demo_val_size = 10000
