from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
from transformers import Trainer
from src.stt.whisper.whisper_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.adapters.huggingface import *
from dotenv import load_dotenv


class WhisperArchitecture():
    def __init__(self, model_id, device, torch_dtype):
        load_dotenv("src/local_settings.env")

        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype

        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device, dtype=self.torch_dtype)
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(self.processor)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps=True,
        )


    def transcribe(self, sample):
        result = self.pipe(sample)

        formatted_chunks = [
            f"[{chunk['timestamp'][0]:.2f}:{chunk['timestamp'][1]:.2f}] {chunk['text']}"
            for chunk in result["chunks"]
        ]

        return "\n".join(formatted_chunks)


    def setup_model_for_train(self, model, processor):
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.eos_token_id

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj"],
            task_type=None
        )
        return get_peft_model(model, config)


    def train(self, dataset, training_args):
        model = self.setup_model_for_train(self.model, self.processor)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.data_collator
        )
        trainer.train()

        trainer.push_to_hub()
        self.processor.push_to_hub(get_repo_name())


