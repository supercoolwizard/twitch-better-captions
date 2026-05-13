from src.services.transcription_service import TranscriptionService
from src.stt.student import StudentModel
from src.stt.teacher import TeacherModel
from src.stt.student_distilled import StudentModelDistilled
from src.config import settings


student = StudentModel(settings.student_model, settings.device)
service = TranscriptionService(student)
service.run()
