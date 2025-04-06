import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler  # pyright: ignore
from tqdm import tqdm
import argparse
import os
import sys
import wandb

# Add parent directory to path for importing models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.deltanet.deltanet import DeltaNetConfig, DeltaNetModel


def count_parameters(model):
    """Count the number of trainable parameters in a model (in millions)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DeltaNet model")
    parser.add_argument(
        "--dataset", type=str, default="wikitext", help="Dataset to use"
    )
    parser.add_argument(
        "--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2", help="Tokenizer to use"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="deltanet",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        help="Weights & Biases mode (online, offline, disabled)",
    )
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging completely",
        default=False,
    )

    return parser.parse_args()


def tokenize_function(examples, tokenizer, seq_length):
    result = tokenizer(
        " ".join(examples["text"]), truncation=False, return_tensors="pt"
    )

    total_length = result["input_ids"].size(1)
    result = {k: v[0] for k, v in result.items()}

    chunks = []
    for i in range(0, total_length - seq_length + 1, seq_length // 2):  # 50% overlap
        chunk = {
            "input_ids": result["input_ids"][i : i + seq_length].clone(),
            "attention_mask": result["attention_mask"][i : i + seq_length].clone()
            if "attention_mask" in result
            else None,
        }
        chunks.append(chunk)

    return chunks


def main():
    args = parse_args()

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_run_name = (
            args.wandb_run_name
            or f"deltanet-h{args.hidden_size}-l{args.num_layers}-heads{args.num_heads}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            mode=args.wandb_mode,
            config={
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "tokenizer": args.tokenizer,
                "batch_size": args.batch_size,
                "seq_length": args.seq_length,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "hidden_size": args.hidden_size,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "architecture": "DeltaNet",
            },
        )

    dataset = load_dataset(args.dataset, args.dataset_config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    def process_dataset(examples):
        return tokenize_function(examples, tokenizer, args.seq_length)

    tokenized_datasets = {}
    for split in dataset:
        chunks = []
        for example in dataset[split]:  # pyright: ignore
            chunks.extend(process_dataset(example))

        tokenized_datasets[split] = chunks

    config = DeltaNetConfig(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_hidden_layers=args.num_layers,
        vocab_size=len(tokenizer),
        max_position_embeddings=args.seq_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id
        if hasattr(tokenizer, "bos_token_id")
        else tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = DeltaNetModel(config)

    num_params = count_parameters(model)
    print(f"Model has {num_params:.2f}M trainable parameters")
    if use_wandb:
        wandb.log({"model/parameters": num_params})

    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "labels": torch.stack([x["input_ids"] for x in batch]),
        },
    )

    val_loader = DataLoader(
        tokenized_datasets["validation"]
        if "validation" in tokenized_datasets
        else tokenized_datasets["test"],
        batch_size=args.batch_size,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "labels": torch.stack([x["input_ids"] for x in batch]),
        },
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps * 0.1,
        num_training_steps=num_training_steps,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    global_step = 0
    best_val_loss = float("inf")

    print(f"Starting training on {device}")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            loss, _, _ = model(input_ids=input_ids, labels=labels)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_train_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({"train_loss": loss.item()})

            if use_wandb and batch_idx % args.log_every == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch + batch_idx / len(train_loader),
                        "train/global_step": global_step,
                    }
                )

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                loss, _, _ = model(input_ids=input_ids, labels=labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_train_loss,
                    "validation/loss": avg_val_loss,
                    "validation/perplexity": torch.exp(
                        torch.tensor(avg_val_loss)
                    ).item(),
                }
            )

    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
