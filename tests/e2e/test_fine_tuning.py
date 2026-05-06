from src.config import settings
from transformers import MoonshineForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Audio


model = MoonshineForConditionalGeneration.from_pretrained(settings.student_model)
processor = AutoProcessor.from_pretrained(settings.student_model)

processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(processor.tokenizer))
model.config.pad_token_id = processor.tokenizer.pad_token_id


