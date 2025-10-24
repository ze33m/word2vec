FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords')"

COPY . . 

RUN mkdir -p models

CMD ["bash", "-c", "python train.py && python upload.py"]
