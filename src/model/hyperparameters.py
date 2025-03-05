hyperparams = {
    # tokenizer
    'vocab_size': 50304,

    # pre-processing
    'batch_size': 16,#500_000,
    'ctx_length': 32, #2048
    'd_model': 48,

    # multi-head attention
    'n_layers': 3,
    'n_heads': 6,
    'head_size': 8,

    # learnig rate
    'lr': 3e-4,  #6e-4

    # other 
    'dropout': 0.0,
    'steps': 1000
}