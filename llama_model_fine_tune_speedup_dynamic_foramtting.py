# to run this script: python3 -m training.llama_model_fine_tune_speedup_dynamic_formating
'''
This script is used to fine-tune the Llama-3.2-1B model on the instruction dataset.
The script is designed to be run on a single node with multiple GPUs.

The script also uses the Hugging Face Hub API to upload the trained model to the Hugging Face Model Hub.
!!! Hugging Face authentication token and other configuration parameters are defined in the config.py.
!!! The model parameters and training parameters are defined in the training.parameters module. 

!!! The model parameters and training parameters are defined in the training.parameters module.

Please define the following environment variables before running this script:
- finetuning_logfile_path: Path to save the finetuning log file
    !!! adjust this logfile path in the llama_pretraining_speedup_utils.py file as well to match this path.
- tokenizer_file_path: Path to the tokenizer file
'''

finetuning_logfile_path = "training/GQA_finetuning.log"
tokenizer_file_path = "Llama-3.2-1B/original/tokenizer.model"

import json
import os
import sqlite3
import sys
import time
from typing import Any, List, Optional
import torch
from functools import partial
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from matplotlib import pyplot as plt
from tqdm import tqdm
from dataPreparation.data_preprocessing_utils import fetch_dataset_from_hf
from training.llama_prompt import InstructionPrompt
from training.parameters import LLAMA32_CONFIG_1B, training_parameters
from training.llama_structure import Llama3Model
from config import (
                        hf_token, 
                        base_model_hf_repo_id,
                        GQA_base_model_path,
                        finetuning_data_hf_repo_id, 
                        finetuning_data_hf_filename, 
                        finetuned_model_hf_repo_id, 
                        finetuned_GQA_model_path,
                        finetuning_GQA_loss_plot_path)
from training.llama_training_speedup_utils import (
                        Tokenizer,  
                        LlamaDataset,
                        create_dataloader,
                        custom_collate_fn,
                        rescale_theta, 
                        train_model, 
                        plot_losses, 
                        fetch_model_from_hf,
                        upload_file_to_hf,
                        EarlyStopping,
                        )


# Set environment variables for distributed training
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(finetuning_logfile_path),
                        logging.StreamHandler()
                    ])
logging.info(f"Model Configuration: {LLAMA32_CONFIG_1B}")
logging.info(f"Training Parameters: {training_parameters}")

def generate_combined_training_prompts_from_db(
        db_path: str,
        tokenizer: Any,
        system_prompt: str = "You are a helpful assistant",
        table_names: Optional[List[str]] = None,
        exclude_tables: Optional[List[str]] = None
    ) -> str:
    """
    Generate combined training prompts from an SQLite database.
    Args:
        db_path (str): Path to the SQLite database file.
        tokenizer (Any): Tokenizer instance for encoding text.
        system_prompt (str): System prompt for formatting training data.
        table_names (Optional[List[str]]): Specific tables to process. If None, process all tables.
        exclude_tables (Optional[List[str]]): Tables to exclude from processing.
    Returns:
        str: Combined training prompts as a single string.
    """
    if not os.path.exists(db_path):
        print(f"❌ DEBUG: Database file not found: {db_path}")
        return ""

    exclude_tables = exclude_tables or []
    prompt_builder = InstructionPrompt(system_prompt=system_prompt)
    all_prompts = []
    total_tokens = 0
    total_rows = 0

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Determine which tables to use
        if table_names:
            tables = [t for t in table_names if t not in exclude_tables]
        else:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            all_db_tables = [row[0] for row in cursor.fetchall()]
            tables = [t for t in all_db_tables if t not in exclude_tables and not t.startswith("sqlite_")]

        if not tables:
            print("⚠️ DEBUG: No valid tables to process.")
            return ""

        for table in tables:
            print(f"🔄 DEBUG: Processing table: {table}")
            try:
                cursor.execute(f"SELECT json_blob FROM {table}")
                rows = cursor.fetchall()

                if not rows:
                    print(f"⚠️ DEBUG: No data in table: {table}")
                    continue

                for row in tqdm(rows, desc=f"Processing '{table}'", unit="row"):
                    try:
                        conversation = json.loads(row[0])
                        # user_msg = next(
                        #     msg["content"] for msg in conversation
                        #     if msg["role"] != "assistant"
                        # )
                        # assistant_msg = next(
                        #     msg["content"] for msg in conversation
                        #     if msg["role"] == "assistant"
                        # )

                        # Currently normalize the conversation to have "user" and "assistant" roles
                        normalized_conversation = [
                                {"role": "user" if msg["role"] != "assistant" else "assistant", "content": msg["content"]}
                                for msg in conversation
                            ]
                        user_msg = next(msg["content"] for msg in normalized_conversation if msg["role"] == "user")
                        assistant_msg = next(msg["content"] for msg in normalized_conversation if msg["role"] == "assistant")

                        prompt = prompt_builder.generate_training_prompt(user_msg, assistant_msg)
                        all_prompts.append(prompt)

                        tokens = tokenizer.encode(prompt) 
                        total_tokens += len(tokens)
                        total_rows += 1

                    except Exception as e:
                        print(f"⚠️ DEBUG: Skipping malformed row in '{table}': {e}")

            except Exception as e:
                print(f"❌ DEBUG: Error reading table '{table}': {e}")
    finally:
        conn.close()

    if not all_prompts:
        print("⚠️ DEBUG: No valid prompts generated.")
        return ""

    print(f"\n✅ DEBUG: Total rows processed: {total_rows}")
    print(f"🧠 DEBUG: Total tokens estimated: {total_tokens:,}")

    return " ".join(all_prompts)



def main(rank, world_size):
    """
    Main function for distributed fine-tuning of the Llama model.
    Args:
        rank (int): Rank of the current process.
        world_size (int): Total number of processes (GPUs).
    """
    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    logging.info(f"[Process {rank}] Using device {device}")

    tokenizer = Tokenizer(tokenizer_file_path)
   
    # Fetch the instruction finetuning data from the Hugging Face Hub
    local_db_path = fetch_dataset_from_hf(hf_token, finetuning_data_hf_repo_id, finetuning_data_hf_filename)
    system_prompt = "You are a helpful and professional customer service agent for a property rental company. Your job is to answer client inquiries politely, clearly, and in a friendly, human-like manner."
    system_prompt_general = "You are a thoughtful assistant. For every task, think step-by-step before answering. Identify the goal, plan the steps, explain your reasoning briefly, then give the answer. Always prioritize clarity and logical thinking. If something is unclear, ask for clarification first."
    data = generate_combined_training_prompts_from_db(
                                db_path= local_db_path,
                                tokenizer = tokenizer,  
                                system_prompt= system_prompt,
                                exclude_tables=["general_instruction_data"]
                                )
    general_data = generate_combined_training_prompts_from_db(
                                db_path= local_db_path,
                                tokenizer = tokenizer,  
                                system_prompt= system_prompt_general,
                                table_names=["general_instruction_data"]
                                )
    data += general_data
    if rank == 0:
        logging.info(f"[Process {rank}] Number of records: {len(data)}")

    if not data:
        logging.info(f"[Process {rank}] No content found in the database.")
        return
    
    # Split dataset for training and validation
    train_ratio = 0.90
    split_idx = int(train_ratio * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    old_context_length = LLAMA32_CONFIG_1B["context_length"]
    LLAMA32_CONFIG_1B["context_length"] = 1024
    LLAMA32_CONFIG_1B["rope_base"] = rescale_theta(
            LLAMA32_CONFIG_1B["rope_base"], old_context_length, LLAMA32_CONFIG_1B["context_length"]
        )
    
    batch_size = training_parameters["batch_size"]
    stride = training_parameters["stride"]
    num_epochs = training_parameters["num_epochs"]
    max_length = LLAMA32_CONFIG_1B["context_length"]

    # Create dataset objects for train and validation data
    train_dataset = LlamaDataset(train_data, tokenizer, max_length, stride)
    val_dataset = LlamaDataset(val_data, tokenizer, max_length, stride)

    # Create Distributed Samplers using the datasets
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    logging.info(f"[Process {rank}] train_dataset length: {len(train_dataset)}")
    logging.info(f"[Process {rank}] val_dataset length: {len(val_dataset)}")
    logging.info(f"[Process {rank}] train_sampler has {len(train_sampler)} samples")
    logging.info(f"[Process {rank}] val_sampler has {len(val_sampler)} samples")

    # Create DataLoaders using the create_dataloader function
    # Note: We now pass the dataset objects rather than raw text.
    train_loader = create_dataloader(
        train_dataset, tokenizer, batch_size, max_length, stride, shuffle=False, drop_last=True, sampler=train_sampler
    )
    val_loader = create_dataloader(
        val_dataset, tokenizer, batch_size, max_length, stride, shuffle=False, drop_last=False, sampler=val_sampler
    )

    # Debug: Retrieve one batch to inspect dimensions
    for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
        logging.info(f"[Process {rank}] Debug: Batch {batch_idx} shapes - input: {input_batch.shape}, target: {target_batch.shape}")
        break

    # Load the model from the Hugging Face Hub
    local_model_filepath = fetch_model_from_hf(hf_token, base_model_hf_repo_id, GQA_base_model_path)
    checkpoint = torch.load(local_model_filepath, map_location="cpu", weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model = Llama3Model(LLAMA32_CONFIG_1B).to(device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name].to(device))
            else:
                logging.info(f"Warning: {name} not found in state_dict.")

    #model = torch.compile(model)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1, fused=True)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(training_parameters["warmup_steps"] * total_steps)

    # Start training timer
    start_time = time.time()

    # Reset sampler for each epoch to ensure proper shuffling
    train_sampler.set_epoch(0)


    early_stopping = EarlyStopping( training_parameters["early_stopping_patience"], training_parameters["early_stopping_delta"], training_parameters["use_early_stopping"], "finetune_checkpoint.pth")
    # Call train_model function once
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_loader, val_loader, optimizer, device, n_epochs=num_epochs,
        eval_freq=training_parameters["eval_freq"],
        eval_iter=training_parameters["eval_iter"],
        start_context=training_parameters["start_context"],
        context_length=max_length,
        tokenizer=tokenizer, warmup_steps=warmup_steps,
        initial_lr=training_parameters["initial_lr"],
        min_lr=training_parameters["min_lr"],
        log_file= finetuning_logfile_path,
        early_stopping=early_stopping

    )
    logging.info(f"[Process {rank}] Training completed.")

    elapsed_time = time.time() - start_time
    logging.info(f"[Process {rank}] Training completed in {elapsed_time:.2f} seconds.")

    # Synchronize all processes before saving
    dist.barrier()
    if rank == 0:
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        plt.savefig(finetuning_GQA_loss_plot_path)
        print("Plot saved")
        logging.info(f"[Process {rank}] Finetuninng Loss plot saved in {finetuning_GQA_loss_plot_path}.")
        response = upload_file_to_hf(hf_token, finetuned_model_hf_repo_id, finetuning_GQA_loss_plot_path)
        logging.info(f"[Process {rank}] Finetuned Model Loss Plot uploaded to Hugging Face Model Hub.")
        logging.info(f"[Process {rank}] Finetuned Model Loss Plot Hub Response: {response}")

        # Save model and optimizer state
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, finetuned_GQA_model_path)
        logging.info(f"[Process {rank}] Finetuned Model saved. Total Training Time: {elapsed_time:.2f} seconds\n")

        # Upload model to Hugging Face Model Hub
        response = upload_file_to_hf(hf_token, finetuned_model_hf_repo_id, finetuned_GQA_model_path)
        logging.info(f"[Process {rank}] Finetuned Model uploaded to Hugging Face Model Hub.")
        logging.info(f"[Process {rank}] Finetuned Model Hub Response: {response}")
     
    # Cleanup
    dist.destroy_process_group()

##########################################################################
# Main entry point
##########################################################################
if __name__ == "__main__":
    # """
    # Entry point for multi-GPU training. Spawns processes for each GPU.
    # """
    world_size = torch.cuda.device_count()  # Get number of GPUs available
    if world_size < 2:
        logging.warning("Warning: At least 2 GPUs are recommended for multi-GPU training.")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
