from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperForConditionalGeneration, pipeline
from transformers import WhisperProcessor
from peft import LoraConfig, get_peft_model
from transformers import Trainer
from transformers.data import data_collator
from src.stt.providers.base_stt import STTModel
from transformers import Seq2SeqTrainingArguments
from src.stt.collators.whisper_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.adapters.huggingface import *
from peft import PeftModel


class WhisperArchitecture(STTModel):
    def __init__(self, model_id, device, torch_dtype):
        super().__init__(model_id, device, torch_dtype)

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


    def prepare_inputs(self, sample):
        inputs = self.processor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        ).to(self.device, dtype=self.torch_dtype)
        return inputs


    def _prepare_sample(self, batch):
        sample = batch["audio"]
        inputs = self.prepare_inputs(sample)
        batch["input_features"] = inputs.input_features[0]

        with open(batch["teacher_path"], "r", encoding="utf-8") as f:
            transcript = f.read().strip()

        batch["labels"] = self.processor.tokenizer(transcript).input_ids
        return batch


    def chunk_audio(self, audio, sr, chunk_s=30):
        chunk_size = sr * chunk_s
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i+chunk_size]

    ### transcription using pipeline automatic chunking
    def transcribe(self, sample):
        result = self.pipe(sample)

        full_text = " ".join(
            chunk["text"] for chunk in result["chunks"]
        )

        return full_text

    ### transcription using chunks of 30s
    # def transcribe(self, sample):
    #     pipe = pipeline(
    #         "automatic-speech-recognition",
    #         model=self.model,
    #         tokenizer=self.processor.tokenizer,
    #         feature_extractor=self.processor.feature_extractor,
    #         chunk_length_s=30,
    #         batch_size=16, 
    #         torch_dtype=self.torch_dtype,
    #         device=self.device,
    #     )
    #     result = pipe(sample)
    #     return result["text"]

    ### transcription using native .gnerate method of the model
    # def transcribe(self, sample):
    #     inputs = self.prepare_inputs(sample)
    #     pred_ids = self.model.generate(
    #         **inputs,
    #         task="transcribe",
    #         language="en",
    #         return_timestamps=True
    #     )
    #     pred_text = self.processor.batch_decode(pred_ids)
    #     pred_text = pred_text[0]
    #
    #     return pred_text


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


    def train(self, dataset):
        model = self.setup_model_for_train(self.model, self.processor)
        dataset = dataset.map(self._prepare_sample, remove_columns=list(dataset.column_names))

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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.data_collator
        )
        trainer.train()
        trainer.push_to_hub()
        self.processor.push_to_hub(get_repo_name())


