import datasets
from datasets import load_dataset, Audio
from transformers import (
    MoonshineForConditionalGeneration, 
    AutoProcessor, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
from src.fine_tuning.collator import DataCollatorSpeechSeq2SeqWithPadding


class StudentTrainer:
    def __init__(self, settings, hf_token, hf_model_repo):
        self.settings = settings
        self.hf_token = hf_token
        self.hf_model_repo = hf_model_repo
        self.processor = AutoProcessor.from_pretrained(settings.student_model)
        self.model = self._setup_model()
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)


    def _setup_model(self):
        model = MoonshineForConditionalGeneration.from_pretrained(self.settings.student_model)

        self.processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(self.processor.tokenizer))
        model.config.pad_token_id = self.processor.tokenizer.pad_token_id

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj"],
            task_type=None
        )
        return get_peft_model(model, config)


    def _prepare_sample(self, batch):
        audio = batch["audio_path"]
        inputs = self.processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        )

        batch["input_values"] = inputs.input_values[0]

        with open(batch["teacher_path"], "r", encoding="utf-8") as f:
            transcript = f.read().strip()

        batch["labels"] = self.processor.tokenizer(transcript).input_ids
        return batch


    def load_and_preprocess_data(self):
        dataset = load_dataset("csv", data_files=str(self.settings.DATA_DIR / "train" / "metadata.csv"))
        dataset = dataset.cast_column("audio_path", datasets.Value("string"))
        dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
        dataset = dataset.map(self._prepare_sample, remove_columns=list(dataset["train"].column_names))

        return dataset["train"]


    def train(self, train_dataset):
        training_args = TrainingArguments(
            output_dir=str(self.settings.MODELS_DIR),
            push_to_hub=True,
            hub_model_id=self.hf_model_repo,
            hub_token=self.hf_token,
            hub_strategy="every_save",

            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            dataloader_pin_memory=False,

            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.0,
            max_grad_norm=1.0,

            bf16=True,

            save_strategy="steps",
            save_steps=200,
            logging_steps=20,

            remove_unused_columns=False,
            label_names=["labels"],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self.data_collator,
        )

        trainer.train()
        # self.model.save_pretrained(str(self.settings.MODELS_DIR))
