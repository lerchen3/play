import argparse
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq


def prepare_dataset(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    return {'train': train_dataset, 'test': test_dataset}


def build_model(model_path: str, seq_len: int):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=seq_len,
        trust_remote_code=True,
        load_in_4bit=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def main(args: argparse.Namespace) -> None:
    dataset = prepare_dataset(args.sft_csv)
    model, tokenizer = build_model(args.model_path, args.seq_len)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        dataset_text_field="text",
        max_seq_length=args.seq_len,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_acc,
            num_train_epochs=args.epochs,
            learning_rate=3e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained_merged(args.output_dir, tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune on the SFT dataset")
    parser.add_argument("--sft_csv", required=True, help="Dataset from compare_and_build_sft_dataset.py")
    parser.add_argument("--model_path", default="unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    parser.add_argument("--output_dir", default="formalization_model")
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)
