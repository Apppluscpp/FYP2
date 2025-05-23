# ✅ updated log and tokenizer paths
finetuning_logfile_path = "training/GQA_finetuning_rlrefined.log"
tokenizer_file_path = "Llama-3.2-1B/original/tokenizer.model"
filtered_data_path = "training/filtered_reinforced_data.txt"

import os, torch, time, logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import pyplot as plt
from training.parameters import LLAMA32_CONFIG_1B, training_parameters
from training.llama_structure import Llama3Model
from training.llama_training_speedup_utils import (
    Tokenizer, InstructionDataset, custom_collate_fn,
    train_model, plot_losses, EarlyStopping, upload_file_to_hf,
    rescale_theta, load_filtered_txt_data
)
from config import (
    hf_token,
    finetuned_model_hf_repo_id,
    finetuning_GQA_loss_plot_path,  # Optional: rename
)

# ✅ updated output model path
output_model_filename = "model/instruction_fine_tuned_llama32_rlrefined.pth"
output_loss_plot = "training/finetuning_loss_rlrefined.png"
previous_model_path = "model/instruction_fine_tuned_llama32_conversation2.pth"

# Setup logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(finetuning_logfile_path), logging.StreamHandler()])

def main(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tokenizer = Tokenizer(tokenizer_file_path)

    logging.info(f"[{rank}] Loading RL-filtered data from: {filtered_data_path}")
    data = load_filtered_txt_data(filtered_data_path)
    print(f"📦 Total entries loaded from file: {len(data)}")

    train_ratio = 0.9
    split_idx = int(train_ratio * len(data))
    raw_train = data[:split_idx]
    raw_val = data[split_idx:]

    print(f"🟢 Initial Training set size: {len(raw_train)}")
    print(f"🔵 Initial Validation set size: {len(raw_val)}")

    max_length = LLAMA32_CONFIG_1B["context_length"] = 1024
    LLAMA32_CONFIG_1B["rope_base"] = rescale_theta(
        LLAMA32_CONFIG_1B["rope_base"], 2048, max_length)

    # Filter short sequences
    min_required_length = 128
    train_data = [d for d in raw_train if len(tokenizer.encode(d)) >= min_required_length]
    val_data = [d for d in raw_val if len(tokenizer.encode(d)) >= min_required_length]

    print(f"🟢 Filtered Training set size: {len(raw_train)}")
    print(f"🔵 Filtered Validation set size: {len(raw_val)}")

    # Dataset and loaders
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    batch_size = training_parameters["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                              sampler=train_sampler, collate_fn=lambda x: custom_collate_fn(
                                  x, tokenizer.pad_token_id, -100, max_length, device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                            sampler=val_sampler, collate_fn=lambda x: custom_collate_fn(
                                x, tokenizer.pad_token_id, -100, max_length, device))

    # ✅ Load previous checkpoint
    checkpoint = torch.load(previous_model_path, map_location="cpu")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model = Llama3Model(LLAMA32_CONFIG_1B).to(device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name].to(device))
    model = model.to(torch.bfloat16)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1, fused=True)

    num_epochs = training_parameters["num_epochs"]
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(training_parameters["warmup_steps"] * total_steps)
    train_sampler.set_epoch(0)

    early_stopping = EarlyStopping(
        training_parameters["early_stopping_patience"],
        training_parameters["early_stopping_delta"],
        training_parameters["use_early_stopping"],
        "finetune_checkpoint_rlrefined.pth"
    )

    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_loader, val_loader, optimizer, device, num_epochs,
        eval_freq=training_parameters["eval_freq"],
        eval_iter=training_parameters["eval_iter"],
        start_context=training_parameters["start_context"],
        context_length=max_length,
        tokenizer=tokenizer,
        warmup_steps=warmup_steps,
        initial_lr=training_parameters["initial_lr"],
        min_lr=training_parameters["min_lr"],
        log_file=finetuning_logfile_path,
        early_stopping=early_stopping
    )

    dist.barrier()
    if rank == 0:
        torch.save({
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, output_model_filename)

        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

        print(f"Epochs seen: {epochs_tensor.tolist()}")
        print(f"Train losses: {train_losses}")
        print(f"Val losses: {val_losses}")
        print(f"Tokens seen: {tokens_seen}")


        plot_losses(torch.linspace(0, num_epochs, len(train_losses)), tokens_seen, train_losses, val_losses)
        plt.savefig(output_loss_plot)
        upload_file_to_hf(hf_token, finetuned_model_hf_repo_id, output_model_filename)

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
