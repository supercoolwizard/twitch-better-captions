from transformers import Seq2SeqTrainingArguments
from src.config import settings
from src.adapters.huggingface import *

training_args = Seq2SeqTrainingArguments(
    output_dir = str(settings.MODELS_DIR),

    push_to_hub=True,
    hub_model_id=get_repo_name(),
    hub_token=get_hf_token(),
    hub_strategy="every_save",

    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    dataloader_pin_memory=False,

    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    bf16=True,

    num_train_epochs=3,

    save_strategy="steps",
    save_steps=100,
    logging_steps=10,

    remove_unused_columns=False,
    label_names=["labels"],
)

