
class TranscriptionService:
    def __init__(self, dataset_loader):
        self.dataset_loader = dataset_loader

    def run(self, stt_instance, role, split):
        dataset = self.dataset_loader.load_dataset(split)
        for i in range(len(dataset)):
            current_audio = dataset[i]["audio"]
            if role == "teacher":
                current_path = dataset[i]["teacher_path"]
            elif role == "student":
                current_path = dataset[i]["student_path"]
            elif role == "student_distilled":
                current_path = dataset[i]["student_distilled_path"]

            current_transcript = stt_instance.transcribe(current_audio)[0]

            with open(current_path, "w") as f:
                f.write(current_transcript)

