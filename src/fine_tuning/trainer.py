from transformers import Trainer, TrainingArguments

def run_distillation_training(model, train_dataset, data_collator, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=30,
        max_steps=100,
        fp16=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    return trainer

