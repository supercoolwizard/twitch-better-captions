from src.config import settings
from transformers import MoonshineForConditionalGeneration, TrainingArguments, Trainer
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Audio
from src.fine_tuning.collator import DataCollatorSpeechSeq2SeqWithPadding
import datasets

model = MoonshineForConditionalGeneration.from_pretrained(settings.student_model)
processor = AutoProcessor.from_pretrained(settings.student_model)

processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(processor.tokenizer))
model.config.pad_token_id = processor.tokenizer.pad_token_id

# model
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],
    task_type=None
)

model = get_peft_model(model, config)

# data
dataset = load_dataset("csv", data_files={
    "train": str(settings.DATA_DIR / "train" / "metadata.csv"),
    "test": str(settings.DATA_DIR / "test" / "metadata.csv")
})
dataset = dataset.cast_column("audio_path", datasets.Value("string"))
dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    audio = batch["audio_path"]
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    )
    # print(inputs.input_values[0])
    batch["input_values"] = inputs.input_values[0]

    with open(batch["teacher_path"], "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    batch["labels"] = processor.tokenizer(transcript).input_ids
    return batch

dataset = dataset.map(
    prepare_dataset,
    remove_columns=dataset["train"].column_names
)
# print(dataset["train"])

# fine_tuning
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

training_args = TrainingArguments(
    output_dir=str(settings.MODELS_DIR),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=50,
    max_steps=500,
    fp16=False,
    remove_unused_columns=False,
    label_names=["labels"],
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(str(settings.MODELS_DIR))
