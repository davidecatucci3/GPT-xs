hyperparams = {
    # tokenizer
    'vocab_size': 50304,

    # pre-processing
    'batch_size': 64,#500_000,
    'ctx_length': 1024, #2048
    'd_model': 768,

    # multi-head attention
    'n_layers': 12,
    'n_heads': 12,
    'head_size': 64,

    # learnig rate
    'lr': 3e-4,  #6e-4

    # other 
    'dropout': 0.0,
    'steps': 1000
}