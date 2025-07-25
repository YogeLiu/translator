import torch


class Config:
    # Model parameters
    d_model = 256
    n_heads = 8
    n_layers = 1
    d_ff = 1024
    max_seq_len = 256
    dropout = 0.1

    # Training parameters
    batch_size = 16
    learning_rate = 5e-4
    num_epochs = 100
    warmup_steps = 500
    label_smoothing = 0.1

    # Learning rate scheduling
    use_cosine_scheduler = True
    cosine_restart_period = 10
    lr_min = 1e-6

    # Gradient accumulation
    gradient_accumulation_steps = 2

    # Early stopping
    patience = 50
    min_delta = 0.001

    # Data parameters
    dataset_name = "YogeLiu/zh-en-translation-dataset-60K"
    src_lang = "zh"
    tgt_lang = "en"
    src_vocab_size = 10000
    tgt_vocab_size = 8000

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
    is_demo = False
    demo_train_size = 100000
    demo_val_size = 50000

    resume_checkpoint = "./model/best_model.pt"  # 设置为None表示不 使用预训练模型，设置为路径表示加载该路径的模型继续训练
