hyperparams = {
    # tokenizer
    'vocab_size': 50304,

    # pre-processing
    'batch_size': 32,#500_000,
    'ctx_length': 32, #2048
    'd_model': 20,

    # multi-head attention
    'n_layers': 2,
    'n_heads': 5,
    'head_size': 4,

    # learnig rate
    'lr': 3e-4,  #6e-4

    # other 
    'dropout': 0.0,
    'steps': 1000
}