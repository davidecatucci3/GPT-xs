hyperparams = {
    # tokenizer
    'vocab_size': 50304,

    # pre-processing
    'batch_size': 16,#500_000,
    'ctx_length': 32,#2048,
    'd_model': 64,#768,

    # multi-head attention
    'n_layers': 4,#12,
    'n_heads': 4,#12,
    'head_size': 16, #64,

    # learnig rate
    'lr': 1e-2,#6e-4

    # other 
    'dropout': 0.2
}