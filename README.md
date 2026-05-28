This software allows you to improve performance of small stt models using black-box distillation method.

First you will need to install dependancies by running pip install -r requirements.txt
as well as pip install -e .

You will also have to add a file in the directory src/local_settings.env with your HF_TOKEN, that you can create here: https://huggingface.co/settings/tokens
HF_TOKEN is used to store transcripts (hf buckets), as well as distilled model (hf repository).

You will also have to add your mp3 files in the input/ folder. As well as configure config.py to your choice (choose teacher/student model)

After this you will have to run an actual script by running bash scripts/run_pipeline.sh, or, by running student_teacher_inference.sh and then distill_student_and_inference.sh, if any part of this process (which is relatively time consuming), I highly encourage you to inspect app/ as it allows you to run each of commands separately using typer wrapper (https://typer.tiangolo.com/#run-it)

More about the methodology used:
This project is inspired by https://arxiv.org/abs/2511.10643 but instead of LLMs, STT models are being used, and no GAN were used. Teacher, and student model are both ran on train/test datasets, then dataloader is created by slicing audio by timestamps teacher model is created, student model is then fine-tuned on this data using LoRA and saved on HF. This, now distilled student is inferenced on test data. Results then can be evaluated and compared to raw student/teacher.
