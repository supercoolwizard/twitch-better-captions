
class TranscriptionService:
    def __init__(self, dataset_loader):
        self.dataset_loader = dataset_loader

    def _match_role_to_path(self, role):
        if role == "teacher":
            return "teacher_path"
        elif role == "student":
            return "student_path"
        elif role == "student_distilled":
            return "student_distilled_path"

    def run(self, stt_instance, role, split):
        dataset = self.dataset_loader.load_dataset(split)

        for sample in dataset:
            audio = sample["audio"]
            output_path = sample[self._match_role_to_path(role)]

            transcript = stt_instance.transcribe(audio)

            with open(output_path, "w") as f:
                f.write(transcript)

