import json
from trainer import MIREXTrainer
import transformers
from mirex_dataset import MIREXCustomDataset
import rwkv_utils
import miditok
from pathlib import Path

import argparse

import torch

torch.set_float32_matmul_precision("medium")


class DebugCallback(transformers.TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        import pdb

        pdb.set_trace()


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--model", type=str)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--val-data", type=Path)

    return parser.parse_args()


def main():
    args = __parse_args()

    if args.train_config:
        with args.train_config.open() as io:
            train_config = json.load(io)
    else:
        train_config = {}

    config = miditok.TokenizerConfig(
        pitch_range=(0, 127),
        use_velocities=False,
        encode_ids_splits="no",
        use_pitchdrum_tokens=False,
    )

    tokenizer = miditok.REMI(config)

    train_data = MIREXCustomDataset(
        args.train_data, tokenizer=tokenizer, max_pitch_offset=6
    )
    if args.val_data:
        val_data = MIREXCustomDataset(args.val_data, tokenizer=tokenizer)
        val_data = torch.utils.data.Subset(val_data, range(4000))
    else:
        print("Running without validation data.")
        val_data = None
        keys = list(train_config.keys())
        for k in keys:
            if "eval" in k:
                train_config.pop(k)

    if args.model.upper() == "RWKV":
        model = rwkv_utils.get_rwkv_model(tokenizer.vocab_size)
    else:
        model_config = transformers.AutoConfig.from_pretrained(
            args.model,
            vocab_size=tokenizer.vocab_size,
            hidden_size=512,
            intermediate_size=1376,
            num_hidden_layers=8,
            num_attention_heads=8,
        )

        model = transformers.AutoModelForCausalLM.from_config(
            model_config, trust_remote_code=True
        )
        model.resize_token_embeddings(tokenizer.vocab_size)

    num_parameters = sum([x.numel() for x in model.parameters()])
    print(f"Model Parameters:\t{num_parameters / 1e6}")

    data_collator = miditok.pytorch_data.DataCollator(tokenizer.pad_token_id)

    train_args = transformers.TrainingArguments(**train_config)

    trainer = MIREXTrainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        # callbacks=[DebugCallback()]
    )

    last_checkpoint = None
    if Path(train_config["output_dir"]).exists():
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(
            train_config["output_dir"]
        )

    if last_checkpoint:
        print(f"Resuming from:\t{last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
