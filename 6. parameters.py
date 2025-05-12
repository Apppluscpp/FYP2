import torch

LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 512,            # NEW: Half the embedding dimension
    "n_heads": 16,              # Number of attention heads
    "n_layers": 4,             # NEW: Half the number of layers
    "hidden_dim": 4096,         # NEW: Almost half the size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to save memory
    "rope_freq": {              # RoPE frequency scaling
        "factor": 32.0,         # NEW: Adjustment of the rescaling factor
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

LLAMA32_IMPROVED_CONFIG = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 512,
    "n_heads": 16,
    "n_layers": 4,
    "hidden_dim": 4096,
    "latent_dim": 16,  # Critical new parameter must be cfg["latent_dim"] <= (cfg["emb_dim"] // cfg["n_heads"]) and even for rope computation
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 2.0,  # Adjusted
        "high_freq_factor": 8.0,  # Adjusted
        "original_context_length": 8192,
    }
}

'''
!!!!!!!!ORIGINAL VALUE!!!!!!!!

LLAMA32_IMPROVED_CONFIG = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "latent_dim": 32,  # Critical new parameter must be cfg["latent_dim"] <= (cfg["emb_dim"] // cfg["n_heads"]) and even for rope computation
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 2.0,  # Adjusted
        "high_freq_factor": 8.0,  # Adjusted
        "original_context_length": 8192,
    }
}

LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 131_072,  # Context length
    "emb_dim": 2048,            # NEW: Half the embedding dimension
    "n_heads": 32,              # Number of attention heads
    "n_layers": 16,             # NEW: Half the number of layers
    "hidden_dim": 8192,         # NEW: Almost half the size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to save memory
    "rope_freq": {              # RoPE frequency scaling
        "factor": 32.0,         # NEW: Adjustment of the rescaling factor
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

'''


lora_rank = 64

training_parameters = {
    "batch_size": 25,
    "stride": 512,
    "num_epochs": 3,
    "eval_freq": 500,
    "eval_iter": 1,
    "start_context": "How to serve customer",
    "initial_lr": 1e-5,
    "min_lr": 1e-5,
    "warmup_steps": 0.2,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0.0
}

ipo_parameters = {
    "batch_size": 20,
    "num_epochs": 3,
    "eval_freq": 100,
    "eval_iter": 1,
    "start_context": "How to serve customer",
    "initial_lr": 1e-6,
    "min_lr": 1e-6,
    "warmup_steps": 0.2,
    "update_interval_pct": 0.3,
    "ema_rate": 0.99,
    "beta": 0.5, 
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_delta": 0.0
}

dpo_parameter = {
    'batch_size': 3,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'num_epochs': 3,
    'max_length': 1024, # Maximum length of the input sequence , will have padding 
    'beta': 0.2,
    'grad_accum_steps': 1,
    }

llama_generation_parameters = {
    "max_new_tokens": 1024,
    "eos_id": [128001, 128009],
    "temperature": 0.5,
    "top_k": 25,
    "frequency_penalty":0.0,
    "presence_penalty": 0.0
}




