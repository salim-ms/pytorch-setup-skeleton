FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

COPY . .
RUN pip install -r requirements.txt
