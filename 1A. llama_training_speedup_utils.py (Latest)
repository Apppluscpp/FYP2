import torch
import torch.nn as nn
import os
import tiktoken
import time
import math
from pathlib import Path
import matplotlib.pyplot as plt
import torch.distributed as dist
from tiktoken.load import load_tiktoken_bpe
from torch.utils.data import Dataset, DataLoader
from matplotlib.ticker import MaxNLocator
import logging
from huggingface_hub import login, hf_hub_download, HfApi
from typing import List

# logfile_path = "training/lora_finetuning.log"
# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     handlers=[
#                         logging.FileHandler(logfile_path),
#                         logging.StreamHandler()
#                     ])


logger = logging.getLogger(__name__)

class Tokenizer:
    """
    Tokenizer class for encoding and decoding text using the LLaMA model.

    Purpose: Handles text tokenization by encoding text into token IDs and decoding token IDs back into text.

    Input parameters:
    - model_path (str): Path to the model file.
      Example: "path/to/model"

    Outputs:
    - None (Initialization of tokenizer with model-specific configurations)
    """
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(model_path)

        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.pad_token_id = self.special_tokens["<|end_of_text|>"]
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens
        )


    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        """
        Encodes input text into token IDs with optional beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens.

        Purpose: Converts the input text into a sequence of tokens that the model can understand.

        Input parameters:
        - text (str): The input text to encode.
          Example: "Hello, world!"
        - bos (bool): Whether to include the beginning-of-sequence token.
          Example: True
        - eos (bool): Whether to include the end-of-sequence token.
          Example: True
        - allowed_special (set): Set of allowed special tokens.
        - disallowed_special (tuple): Tuple of disallowed special tokens.

        Outputs:
        - tokens (list): List of token IDs representing the input text.
          Example: [128000, 12345, 67890, 128001]
        """
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
          tokens = []

        tokens += self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        """
        Decodes a sequence of token IDs into text.

        Purpose: Converts a list of token IDs back into the corresponding text.

        Input parameters:
        - tokens (list): List of token IDs.
          Example: [128000, 12345, 67890, 128001]

        Outputs:
        - text (str): The decoded text.
          Example: "Hello, world!"
        """
        return self.model.decode(tokens)

def assign(left, right, tensor_name="unknown"):
    """
    Assigns the weights from one tensor to another, checking for shape consistency.

    Purpose: Ensures that the tensor shapes match before assignment.

    Input parameters:
    - left (torch.Tensor): The target tensor.
    - right (torch.Tensor or np.array): The source tensor.
    - tensor_name (str): Name of the tensor to provide an error message.
      Example: "model.layers.0.self_attn.q_proj.weight"

    Outputs:
    - torch.nn.Parameter: A new parameter initialized with the source tensor.
      Example: torch.nn.Parameter(tensor)
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_llama(model, param_config, params):
    """
    Loads pretrained weights into the LLaMA model.

    Purpose: Transfers pretrained model weights from a dictionary to the model layers.

    Input parameters:
    - model (nn.Module): The LLaMA model to load weights into.
    - param_config (dict): Configuration dictionary for the model.
    - params (dict): Dictionary of pretrained model weights.
      Example: {'model.embed_tokens.weight': tensor, ...}

    Outputs:
    - None: The model weights are updated in-place.
    """
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):

        # Load attention weights
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params.keys():
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
        print("Model uses weight tying.")

##########################################################################
# Custom Dataset with debug prints to help identify indexing issues
##########################################################################
class LlamaDataset(Dataset):
    """
    A PyTorch Dataset for training language models like LLaMA on long text data using sliding window chunking.

    This dataset:
    - Tokenizes the input text with support for special tokens.
    - Splits the tokenized text into overlapping chunks of fixed length.
    - Each training example consists of an input sequence and a corresponding target sequence (shifted by one token).

    Args:
        txt (str): Raw text input to be tokenized and chunked.
        tokenizer: A tokenizer object with an `encode()` method compatible with special tokens.
        max_length (int): The maximum length of each input sequence.
        stride (int): The step size to move the sliding window, allowing overlap for better utilization.

    Attributes:
        input_ids (List[torch.Tensor]): List of tokenized input sequences.
        target_ids (List[torch.Tensor]): Corresponding next-token prediction targets.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []  # input sequences
        self.target_ids = []  # target sequences

        # Tokenize the input text
        token_ids = tokenizer.encode(txt, allowed_special={"<|end_of_text|>", "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"})
        
        # Ensure tokenized text is long enough to create chunks
        if len(token_ids) < max_length:
            raise ValueError(f"Text too short to create chunks. Length: {len(token_ids)}, max_length: {max_length}")

        # Create input-target pairs using sliding window
        for i in range(0, len(token_ids) - max_length + 1, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
        logger.info(f"[Dataset] Number of input-target pairs: {len(self.input_ids)}")

    def __len__(self):
        """
        Returns:
            int: Number of input-target pairs available in the dataset.
        """
        total = len(self.input_ids)
        logger.info(f"[Dataset __len__] Total: {total}")
        return total

    def __getitem__(self, idx):
        """
        Fetches the input-target pair at the specified index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The input and corresponding target tensor.

        Raises:
            IndexError: If the index is out of range.
        """
        try:
            return self.input_ids[idx], self.target_ids[idx]
        except IndexError as e:
            logger.error(f"IndexError in __getitem__ for idx: {idx} with dataset length {len(self.input_ids)}")
            raise e

##########################################################################
# Create DataLoader function that supports either raw text or a pre-created dataset
##########################################################################
def create_dataloader(txt_or_dataset, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, sampler=None):
    """
    Creates a PyTorch DataLoader from either raw text or a pre-built Dataset.

    If raw text is provided, this function initializes a `LlamaDataset` instance using 
    the tokenizer and chunking parameters. If a Dataset is passed, it's used directly.

    Args:
        txt_or_dataset (str or Dataset): Raw input text to be tokenized and chunked,
            or an existing Dataset object (e.g., LlamaDataset).
        tokenizer: A tokenizer object with an `encode()` method that supports special tokens.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        max_length (int, optional): Length of each input sequence if using raw text. Defaults to 256.
        stride (int, optional): Step size for the sliding window if using raw text. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data. Ignored if `sampler` is provided. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.
        sampler (Sampler, optional): Custom sampler for loading data. If provided, disables internal shuffling.

    Returns:
        DataLoader: A PyTorch DataLoader instance ready for training or evaluation.
    """
    # If a dataset is provided instead of raw text, use it directly.
    if isinstance(txt_or_dataset, str):
        dataset = LlamaDataset(txt_or_dataset, tokenizer, max_length, stride)
    else:
        dataset = txt_or_dataset

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,  # disable shuffling if a sampler is provided
        drop_last=drop_last,
        sampler=sampler
    )
    return dataloader

def model_memory_size(model, input_dtype=torch.float32):
    """
    Calculate the total memory usage of a model, including parameters, gradients, and buffers.

    Purpose: Provides an estimate of the memory usage of the model, which is helpful for resource planning.

    Input parameters:
    - model (torch.nn.Module): The model whose memory usage is to be calculated.
      Example: model = MyModel()
    - input_dtype (torch.dtype): The data type for the model (default is torch.float32).
      Example: input_dtype = torch.float32

    Outputs:
    - float: The estimated memory usage of the model in gigabytes.
      Example: 1.2 (GB)
    """
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb

def rescale_theta(theta_old, context_length_old, context_length_new):
    """
    Rescale the theta (rope_base) for a new context length.

    Purpose: Adjust the theta to account for changes in the context length when fine-tuning or modifying a model.

    Input parameters:
    - theta_old (float or tensor): The old value of theta (rope_base) before scaling.
      Example: 1.0
    - context_length_old (int): The original context length of the model.
      Example: 1024
    - context_length_new (int): The new context length to scale to.
      Example: 2048

    Outputs:
    - float or tensor: The rescaled theta value.
      Example: 2.0
    """
    scaling_factor = context_length_new / context_length_old
    theta_new = theta_old * scaling_factor
    return theta_new

def text_to_token_ids(text, tokenizer):
    """
    Convert text into token ids using the specified tokenizer.

    Purpose: Encodes text into a sequence of token IDs, which are compatible with the model's tokenizer.

    Input parameters:
    - text (str): The input text to be tokenized.
      Example: "Hello, world!"
    - tokenizer (Tokenizer): The tokenizer instance to encode the text.
      Example: tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    Outputs:
    - torch.Tensor: A tensor containing the token IDs for the input text.
      Example: tensor([[50256, 15496, 11, 50257]])
    """
    encoded = tokenizer.encode(text, allowed_special={"<|end_of_text|>", "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension [1,2,3] --> [[1,2,3]]
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    Convert a sequence of token IDs back into text.

    Purpose: Decodes token IDs into human-readable text.

    Input parameters:
    - token_ids (torch.Tensor): A tensor containing token IDs to decode.
      Example: tensor([[50256, 15496, 11, 50257]])
    - tokenizer (Tokenizer): The tokenizer instance to decode the token IDs.
      Example: tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    Outputs:
    - str: The decoded text corresponding to the token IDs.
      Example: "Hello, world!"
    """
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Generate text using the model in a simple way by predicting the next token at each step.

    Purpose: Generates a sequence of tokens by iteratively predicting the next token based on the current context.

    Input parameters:
    - model (torch.nn.Module): The model used for generating text.
      Example: model = GPT2LMHeadModel.from_pretrained('gpt2')
    - idx (torch.Tensor): The initial input token indices (B, T), where B is the batch size and T is the sequence length.
      Example: tensor([[50256, 15496, 11]])
    - max_new_tokens (int): The number of new tokens to generate.
      Example: 50
    - context_size (int): The maximum size of the context window for the model to consider when generating the next token.
      Example: 1024

    Outputs:
    - torch.Tensor: The generated token IDs as a tensor (B, T + max_new_tokens).
      Example: tensor([[50256, 15496, 11, 50257, 50258, ...]])
    """
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate_and_print_sample(model, tokenizer, device, start_context, context_length):
    """
    Generate text from a model and print the result.

    Purpose: This function generates a sample of text from the model starting from a given context and prints it.

    Input parameters:
    - model (torch.nn.Module): The model used for text generation.
      Example: model = GPT2LMHeadModel.from_pretrained('gpt2')
    - tokenizer (Tokenizer): The tokenizer instance used to encode and decode text.
      Example: tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    - device (torch.device): The device (CPU or GPU) on which the model is running.
      Example: device = torch.device('cuda')
    - start_context (str): The initial text to start the generation from.
      Example: "Once upon a time"
    - context_length (int): The context length (maximum number of tokens) for the model to consider.
      Example: 1024

    Outputs:
    - None: The generated text is printed to the console.
      Example: "Once upon a time, there was a little girl who..."
    """
    model.eval()
    context_size = context_length
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        logger.info(decoded_text.replace("\n", " "))  # Compact log format
    model.train()

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate the model by calculating loss on both the training and validation sets.

    Purpose: Measures the model's performance on training and validation data to track learning progress.

    Input parameters:
    - model (torch.nn.Module): The model to evaluate.
      Example: model = MyModel()
    - train_loader (DataLoader): The DataLoader instance for the training set.
      Example: train_loader = DataLoader(train_data, batch_size=32)
    - val_loader (DataLoader): The DataLoader instance for the validation set.
      Example: val_loader = DataLoader(val_data, batch_size=32)
    - device (torch.device): The device (CPU or GPU) where the model is loaded.
      Example: device = torch.device('cuda')
    - eval_iter (int): The number of iterations to run during evaluation.
      Example: 100

    Outputs:
    - tuple: The average training and validation loss over the evaluation iterations.
      Example: (1.2, 1.1)
    """
    model.eval() #A
    with torch.no_grad(): #B
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculate the loss for a single batch of data.

    Purpose: Computes the loss by comparing the model's predictions with the target batch.

    Input parameters:
    - input_batch (torch.Tensor): The input batch of data.
      Example: input_batch = torch.randn(32, 128)
    - target_batch (torch.Tensor): The target batch of true labels.
      Example: target_batch = torch.randint(0, 50256, (32, 128))
    - model (torch.nn.Module): The model to compute predictions.
      Example: model = MyModel()
    - device (torch.device): The device on which the model is located.
      Example: device = torch.device('cuda')

    Outputs:
    - torch.Tensor: The calculated loss value.
      Example: tensor(2.3)
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculate the average loss for a number of batches from the DataLoader.

    Purpose: Computes the average loss over multiple batches from the data loader for performance evaluation.

    Input parameters:
    - data_loader (DataLoader): The DataLoader instance containing the dataset.
      Example: data_loader = DataLoader(train_data, batch_size=32)
    - model (torch.nn.Module): The model used to compute the loss.
      Example: model = MyModel()
    - device (torch.device): The device (CPU or GPU) on which the model is located.
      Example: device = torch.device('cuda')
    - num_batches (int, optional): The number of batches to process. If None, it will process the entire loader.
      Example: num_batches = 100

    Outputs:
    - float: The average loss over the specified number of batches.
      Example: 1.15
    """
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader) #A
    else:
        num_batches = min(num_batches, len(data_loader)) #B
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item() #C
        else:
            break
    return total_loss / num_batches #D

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, verbose=False, path=None):
        """
        Early stops the training if validation loss doesn't improve after a given patience.

        Parameters:
        - patience (int): How many evaluations to wait after last improvement.
        - delta (float): Minimum change in the monitored loss to qualify as an improvement.
        - verbose (bool): If True, prints updates when early stopping is triggered.
        - path (str): Optional path to save the best model checkpoint.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path

        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model, optimizer):
        """
        This function checks if early stopping should be triggered based on the validation loss.

        Parameters:
        - val_loss (float): The current validation loss.
        - model (nn.Module): The model to save if it is the best one.
        """
        if val_loss < self.best_loss - self.delta:
            # Validation loss improved
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict()
            if self.verbose:
                logger.info(f"Validation loss improved to {val_loss:.3f}. Saving model.")
            if self.path:
                torch.save(self.best_model, self.path)
                # save the model of model state dict and the optimizer state dict
                torch.save({
                    "model_state_dict": self.best_model,
                    "optimizer_state_dict": optimizer.state_dict(),
                }, self.path)
        else:
            # Validation loss did not improve
            self.counter += 1
            if self.verbose:
                logger.info(f"Validation loss did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered. Best validation loss: {self.best_loss:.3f}")

def synchronize_early_stopping(early_stop_flag, device):
    """
    Synchronize the early stopping flag across all processes in a DDP environment.

    Parameters:
    - early_stop_flag (bool): Local early stopping flag.
    - device (torch.device): The device to perform the operation on.

    Returns:
    - bool: Synchronized early stopping flag.
    """
    tensor_flag = torch.tensor(early_stop_flag, dtype=torch.int, device=device)
    torch.distributed.all_reduce(tensor_flag, op=torch.distributed.ReduceOp.SUM)
    return tensor_flag.item() > 0

def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, context_length, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6, log_file="training_log.txt",
                early_stopping=None):
    """
    Trains the model with iteration-based evaluation for monitoring, and supports early stopping 
    based on the validation loss after each epoch to ensure stable and efficient training.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The DataLoader providing training data.
        val_loader (DataLoader): The DataLoader providing validation data.
        optimizer (Optimizer): The optimizer used for updating model parameters.
        device (torch.device): The device (CPU/GPU) used for training.
        n_epochs (int): The number of training epochs.
        eval_freq (int): Frequency (in iterations) to perform evaluation on the validation set.
        eval_iter (int): The number of iterations for evaluation.
        start_context (str): Initial context used to generate samples during training.
        context_length (int): Length of the context for sample generation.
        tokenizer (Tokenizer): The tokenizer used to encode and decode inputs.
        warmup_steps (int): The number of steps for learning rate warmup.
        initial_lr (float, optional): Initial learning rate. Defaults to 3e-05.
        min_lr (float, optional): Minimum learning rate. Defaults to 1e-6.
        log_file (str, optional): File to log training details. Defaults to "training_log.txt".
        early_stopping (EarlyStopping, optional): An instance of the `EarlyStopping` class for stopping training early based on validation loss.

    Returns:
        tuple: 
            - train_losses (list): List of training losses per evaluation step.
            - val_losses (list): List of validation losses per evaluation step.
            - track_tokens_seen (list): List of total tokens seen after each evaluation step.
            - track_lrs (list): List of learning rates used during training.
    """
    if early_stopping and not isinstance(early_stopping, EarlyStopping):
        raise ValueError("The 'early_stopping' parameter must be an instance of the EarlyStopping class.")
    
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_loader) * n_epochs
    logger.info(f"Each Epoch training steps: {len(train_loader)}")
    logger.info(f"Total training steps: {total_training_steps}")

    with open(log_file, "a") as log:
        try:
            for epoch in range(n_epochs):
                model.train()
                epoch_start = time.time()
                
                for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
                    optimizer.zero_grad()
                    global_step += 1

                    # LR scheduling (unchanged)
                    if global_step < warmup_steps:
                        lr = initial_lr + global_step * (peak_lr - initial_lr) / warmup_steps
                    else:
                        progress = (global_step - warmup_steps) / (total_training_steps - warmup_steps)
                        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
                    
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    track_lrs.append(lr)

                    # Forward/backward pass
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    
                    if global_step > warmup_steps:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    tokens_seen += input_batch.numel()

                    # Iteration-based evaluation (kept for monitoring)
                    if global_step % eval_freq == 0:
                        iter_train_loss, iter_val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter
                        )
                        train_losses.append(iter_train_loss)
                        val_losses.append(iter_val_loss)
                        track_tokens_seen.append(tokens_seen)
                        
                        logger.info(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - "
                            f"Epoch {epoch+1}/{n_epochs} | Iter {global_step:06d} | "
                            f"Train Loss: {iter_train_loss:.3f} | "
                            f"Val Loss: {iter_val_loss:.3f} | "
                            f"LR: {lr:.2e}"
                        )

                iter_train_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)[0]
                epoch_val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)[1]

                train_losses.append(iter_train_loss)
                val_losses.append(epoch_val_loss)
                track_tokens_seen.append(tokens_seen)

                if early_stopping:
                    
                    early_stopping(epoch_val_loss, model, optimizer)
                    # Synchronize early stopping across all processes
                    early_stop_flag = synchronize_early_stopping(early_stopping.early_stop, device)
                    if early_stop_flag:
                        logger.info(f"Early stopping at epoch {epoch+1} (best val loss: {early_stopping.best_loss:.3f})")
                        if torch.distributed.get_rank() == 0:  # Only rank 0 restores the best model
                            model.load_state_dict(early_stopping.best_model)
                        break
                
                # Epoch summary
                epoch_time = time.time() - epoch_start
                logger.info(
                    f"Epoch {epoch+1} completed in {epoch_time:.2f}s | "
                    f"Final Train Loss: {iter_train_loss:.3f} | "
                    f"Final Val Loss: {epoch_val_loss:.3f}\n"
                )
                
                generate_and_print_sample(model, tokenizer, device, start_context, context_length)
        finally:
            print("Training completed.")

    return train_losses, val_losses, track_tokens_seen, track_lrs

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plots the training and validation losses over time.

    Purpose: To visualize the training and validation losses during the training process.

    Input parameters:
    - epochs_seen (list): List of epochs seen during training.
      Example: [1, 2, 3, 4, 5]
    - tokens_seen (list): List of tokens seen during training.
      Example: [1000, 2000, 3000, 4000, 5000]
    - train_losses (list): List of training losses for each epoch.
      Example: [0.25, 0.22, 0.18, 0.15, 0.12]
    - val_losses (list): List of validation losses for each epoch.
      Example: [0.30, 0.28, 0.25, 0.23, 0.20]

    Outputs:
    - A plot showing the training and validation losses over time.
    """
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Generates tokens from the model using temperature and top-k sampling.

    Purpose: To generate text sequences based on the given context, using sampling techniques to control diversity.

    Input parameters:
    - model (nn.Module): The neural network model for token generation.
    - idx (Tensor): The input sequence of token IDs to start generation from.
      Example: [101, 2345, 4567]
    - max_new_tokens (int): The maximum number of tokens to generate.
      Example: 50
    - context_size (int): The size of the context to use for generating new tokens.
      Example: 1024
    - temperature (float): Controls the randomness of the generation. Higher values result in more random outputs.
      Example: 1.0
    - top_k (int): Limits the logits to the top-k most likely tokens for sampling.
      Example: 40
    - eos_id (int): The ID of the end-of-sequence token.
      Example: 50256

    Outputs:
    - idx (Tensor): The token IDs of the generated sequence.
      Example: [101, 2345, 4567, 7890, ...]
    """

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def generate_v2(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
    frequency_penalty=0.0,
    presence_penalty=0.0
):
    """
    Generates tokens with frequency and presence penalties, temperature, and top-k sampling.

    Purpose: To generate text sequences that penalize repeated and already seen tokens to improve diversity.

    Input parameters:
    - model (nn.Module): The neural network model for token generation.
    - idx (Tensor): The input sequence of token IDs to start generation from.
      Example: [101, 2345, 4567]
    - max_new_tokens (int): The maximum number of tokens to generate.
      Example: 50
    - context_size (int): The size of the context to use for generating new tokens.
      Example: 1024
    - temperature (float): Controls the randomness of the generation.
      Example: 1.0
    - top_k (int): Limits the logits to the top-k most likely tokens for sampling.
      Example: 40
    - eos_id (list): The IDs of the end-of-sequence token.
      Example: [128001, 128009]
    - frequency_penalty (float): A penalty applied to frequently used tokens.
      Example: 0.5
    - presence_penalty (float): A penalty applied to tokens that are already present in the sequence.
      Example: 0.5

    Outputs:
    - idx (Tensor): The token IDs of the generated sequence.
      Example: [101, 2345, 4567, 7890, ...]
    """
    print(f"Generating text with max_new_tokens={max_new_tokens}, context_size={context_size}, "
          f"temperature={temperature}, top_k={top_k}, eos_id={eos_id}, "
          f"frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}")
    
    # Extract vocab size from the model's token embedding layer
    vocab_size = model.tok_emb.num_embeddings
    
    # Initialize a frequency counter for tokens
    token_freq = torch.zeros((idx.size(0), vocab_size), device=idx.device)

    # For-loop for token generation
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)  # (batch_size, context_len, vocab_size)
        logits = logits[:, -1, :]  # Focus on last timestep logits: (batch_size, vocab_size)

        # Apply frequency and presence penalties
        if frequency_penalty > 0.0 or presence_penalty > 0.0:
            # Update token frequencies for the current sequence
            for batch_idx in range(idx.size(0)):
                unique_tokens, counts = torch.unique(idx[batch_idx], return_counts=True)
                token_freq[batch_idx, unique_tokens] = counts.float()
                # # token_freq[batch_idx, idx_cond[batch_idx, -1]] += 1
                # last_token = idx[batch_idx, -1].item()  # Get last token for this batch
                # token_freq[batch_idx, last_token] += 1
            
            # Compute penalties
            penalties = frequency_penalty * token_freq + presence_penalty * (token_freq > 0).float()
            logits = logits - penalties  # Apply penalties to logits

        # Apply top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Check if all generated tokens belong to eos_id list
        if eos_id and torch.isin(idx_next, torch.tensor(eos_id, device=idx.device)).all():
            break

        # Append generated token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def generate_lora(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
    frequency_penalty=0.0,
    presence_penalty=0.0
):
    """
    Generates tokens with frequency and presence penalties, temperature, and top-k sampling.

    Purpose: To generate text sequences that penalize repeated and already seen tokens to improve diversity.

    Input parameters:
    - model (nn.Module): The neural network model for token generation.
    - idx (Tensor): The input sequence of token IDs to start generation from.
      Example: [101, 2345, 4567]
    - max_new_tokens (int): The maximum number of tokens to generate.
      Example: 50
    - context_size (int): The size of the context to use for generating new tokens.
      Example: 1024
    - temperature (float): Controls the randomness of the generation.
      Example: 1.0
    - top_k (int): Limits the logits to the top-k most likely tokens for sampling.
      Example: 40
    - eos_id (list): The IDs of the end-of-sequence token.
      Example: [128001, 128009]
    - frequency_penalty (float): A penalty applied to frequently used tokens.
      Example: 0.5
    - presence_penalty (float): A penalty applied to tokens that are already present in the sequence.
      Example: 0.5

    Outputs:
    - idx (Tensor): The token IDs of the generated sequence.
      Example: [101, 2345, 4567, 7890, ...]
    """
    print(f"Generating text with max_new_tokens={max_new_tokens}, context_size={context_size}, "
          f"temperature={temperature}, top_k={top_k}, eos_id={eos_id}, "
          f"frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}")
    
    # Extract vocab size from the model's token embedding layer
    vocab_size = model.original_model.tok_emb.num_embeddings
    
    # Initialize a frequency counter for tokens
    token_freq = torch.zeros((idx.size(0), vocab_size), device=idx.device)

    # For-loop for token generation
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)  # (batch_size, context_len, vocab_size)
        logits = logits[:, -1, :]  # Focus on last timestep logits: (batch_size, vocab_size)

        # Apply frequency and presence penalties
        if frequency_penalty > 0.0 or presence_penalty > 0.0:
            # Update token frequencies for the current sequence
            for batch_idx in range(idx.size(0)):
                unique_tokens, counts = torch.unique(idx[batch_idx], return_counts=True)
                token_freq[batch_idx, unique_tokens] = counts.float()
                # # token_freq[batch_idx, idx_cond[batch_idx, -1]] += 1
                # last_token = idx[batch_idx, -1].item()  # Get last token for this batch
                # token_freq[batch_idx, last_token] += 1
            
            # Compute penalties
            penalties = frequency_penalty * token_freq + presence_penalty * (token_freq > 0).float()
            logits = logits - penalties  # Apply penalties to logits

        # Apply top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Check if all generated tokens belong to eos_id list
        if eos_id and torch.isin(idx_next, torch.tensor(eos_id, device=idx.device)).all():
            break
        # Append generated token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
    return idx

def calc_loss_and_perplexity_batch(input_batch, target_batch, model, device):
    """
    Calculates the loss and perplexity for a single batch.

    Purpose: To compute the cross-entropy loss and perplexity for a given batch of inputs and targets.

    Input parameters:
    - input_batch (Tensor): A batch of input token IDs.
      Example: [101, 2345, 4567]
    - target_batch (Tensor): A batch of target token IDs.
      Example: [2345, 4567, 7890]
    - model (nn.Module): The neural network model.
    - device (str): The device (CPU or GPU) to run the model on.

    Outputs:
    - loss (Tensor): The calculated loss for the batch.
      Example: 0.23
    - perplexity (Tensor): The perplexity for the batch.
      Example: 1.26
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    perplexity = torch.exp(loss)
    return loss, perplexity

def calc_loss_and_perplexity_loader(data_loader, model, device, num_batches=None):
    """
    Calculates the average loss and perplexity for a data loader.

    Purpose: To compute the average loss and perplexity over a number of batches in a dataset.

    Input parameters:
    - data_loader (DataLoader): The DataLoader for the dataset.
    - model (nn.Module): The neural network model.
    - device (str): The device (CPU or GPU) to run the model on.
    - num_batches (int): The number of batches to process. Default is None (process all batches).

    Outputs:
    - avg_loss (float): The average loss over the batches.
      Example: 0.25
    - avg_perplexity (float): The average perplexity over the batches.
      Example: 1.22
    """
    total_loss = 0.0
    total_perplexity = 0.0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss, perplexity = calc_loss_and_perplexity_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            total_perplexity += perplexity.item()
        else:
            break
    
    avg_loss = total_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    return avg_loss, avg_perplexity

def plot_losses_and_perplexities(epochs_seen, train_losses, val_losses, train_perplexities, val_perplexities, 
                                 loss_save_path, perplexity_save_path):
    """
    Plot and save the training and validation losses and perplexities.

    Purpose: Visualizes the model's performance during training by plotting the losses and perplexities 
             for both training and validation datasets.

    Input parameters:
    - epochs_seen (list): List of epoch numbers.
      Example: [1, 2, 3, 4]
    - train_losses (list): List of training losses for each epoch.
      Example: [0.5, 0.4, 0.3, 0.2]
    - val_losses (list): List of validation losses for each epoch.
      Example: [0.6, 0.5, 0.4, 0.3]
    - train_perplexities (list): List of training perplexities for each epoch.
      Example: [1.5, 1.3, 1.1, 1.0]
    - val_perplexities (list): List of validation perplexities for each epoch.
      Example: [1.6, 1.4, 1.2, 1.1]
    - loss_save_path (str): Path where the loss plot image will be saved.
      Example: "train_val_loss.png"
    - perplexity_save_path (str): Path where the perplexity plot image will be saved.
      Example: "train_val_perplexity.png"

    Outputs:
    - None: Saves the loss and perplexity plots as images.
    """
    
    # Plot Losses
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs_seen, train_losses, label="Training Loss", color="blue")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation Loss", color="green")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Validation Loss")
    fig.tight_layout()
    
    # Save the loss plot as an image
    plt.savefig(loss_save_path)
    plt.show()

    # Plot Perplexities
    fig, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(epochs_seen, train_perplexities, label="Training Perplexity", color="orange")
    ax2.plot(epochs_seen, val_perplexities, linestyle="--", label="Validation Perplexity", color="red")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Perplexity")
    ax2.legend(loc="upper right")
    ax2.set_title("Training and Validation Perplexity")
    fig.tight_layout()

    # Save the perplexity plot as an image
    plt.savefig(perplexity_save_path)
    plt.show()

def fetch_model_from_hf(hf_token, model_id, filename):
    """
    Logs into Hugging Face, downloads a model file, and returns the local file path.

    Parameters:
    - hf_token (str): Hugging Face API token.
    - model_id (str): Model repository ID.
    - filename (str): Name of the model file.

    Returns:
    - local_file_path (str): Path to the downloaded model file.
    """
    login(token=hf_token)
    local_model_filepath = hf_hub_download(repo_id=model_id, filename=filename, repo_type="model")
    return local_model_filepath

def upload_file_to_hf(hf_token,repo_id, model_path):
    """
    Uploads a model file to a Hugging Face repository.

    Parameters:
    - repo_id (str): Hugging Face repository ID where the model will be stored.
    - model_path (str): Local path to the model file.

    Returns:
    - response: API response after uploading.
    """
    login(token=hf_token)
    api = HfApi()
    
    # Create the repository if it does not exist
    api.create_repo(repo_id=repo_id, exist_ok=True)

    # Upload the model file
    response = api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,  # The file will be stored with the same name in the repo
        repo_id=repo_id
    )

    return response

class InstructionDataset(Dataset):
    """
    Create a dataset for instruction-based tasks where each entry is already fully formatted text.
    
    This dataset is used for tasks where each entry is already formatted with necessary tokens, 
    such as special tokens for instruction, response, and context separation. The text is tokenized 
    into integer sequences using the provided tokenizer.

    Args:
        data (list of str): List of fully formatted strings, each representing an instruction or a task 
                             that is ready for model input.
        tokenizer (Tokenizer): The tokenizer used to encode the text into token IDs. It should support 
                               encoding with special tokens like `<|end_of_text|>`, `<|begin_of_text|>`, 
                               `<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>`.

    Attributes:
        encoded_texts (list of list of int): A list where each element is a tokenized version of 
                                              the corresponding entry from the `data` list.
    
    Methods:
        __getitem__(self, index): Retrieves the tokenized representation of a specific entry by index.
        __len__(self): Returns the total number of entries in the dataset.

    Example:
        data = ["<|begin_of_text|> Instruction: How to make tea?<|end_of_text|>"]
        tokenizer = SomeTokenizer()
        dataset = InstructionDataset(data, tokenizer)
        print(dataset[0])  # This would print the tokenized representation of the first entry.
    """
    def __init__(self, data, tokenizer):
        self.data = data

        self.encoded_texts = []
        for entry in data:
            # full_text is already fully formatted including special tokens
            full_text = entry
            self.encoded_texts.append(
                tokenizer.encode(full_text, allowed_special={"<|end_of_text|>", "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"})
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
    
def custom_collate_fn(
    batch,
    pad_token_id=128001,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Prepare a batch of samples by padding or truncating sequences to ensure uniform context length.
    
    This version assumes that each item in the batch is already pre-formatted with all the necessary
    special tokens (e.g. <|begin_of_text|>, <|end_of_text|>, etc.), so no additional tokens are added.
    
    Inputs:
      - batch (list): List of tokenized sequences (each sequence is a list of token IDs).
      - pad_token_id (int): Token ID used for padding.
      - ignore_index (int): Token value to ignore during loss calculation.
      - allowed_max_length (int or None): Optional maximum length for truncating sequences.
      - device (str): Target device for the output tensors.
    
    Outputs:
      - tuple: (inputs_tensor, targets_tensor) where:
          * inputs_tensor: Tensor of shape (batch_size, seq_len-1)
          * targets_tensor: Tensor of shape (batch_size, seq_len-1) with padding (except for the first padded token) replaced by ignore_index.
    """
    # Find the longest sequence in the batch (no extra tokens are added)
    batch_max_length = max(len(item) for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        # Pad the sequence to the maximum length in the batch
        padded = item + [pad_token_id] * (batch_max_length - len(item))
        
        # Create inputs and targets by shifting the sequence by one
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        
        # Replace all but the first padding token in targets with ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optionally truncate sequences to allowed_max_length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Stack the sequences into tensors and move to the specified device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

def load_filtered_txt_data(file_path: str) -> List[str]:
    """
    Loads prompts formatted like:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    ...
    <|start_header_id|>assistant<|end_header_id|>
    ...
    <|eot_id|><|end_of_text|>
    Returns:
        List of formatted prompt-completion strings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    entries = text.strip().split("<|begin_of_text|>")
    return ["<|begin_of_text|>" + e.strip() for e in entries if e.strip()]
