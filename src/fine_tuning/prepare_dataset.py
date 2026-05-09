
def prepare_dataset(batch, adapter):
    audio = batch["audio_path"]

    processed = adapter.preprocess(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    )
    batch["input_values"] = processed[0]

    with open(batch["teacher_path"], "r", encoding="utf-8") as f:
        batch["labels"] = adapter.processor.tokenizer(f.read().strip()).input_ids
    return batch
