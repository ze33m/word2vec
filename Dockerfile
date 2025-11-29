FROM docker.io/pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords')"

COPY . . 

RUN mkdir -p models

CMD ["bash", "-c", "python download_dataset.py && python convert.py && python upload_vocab.py"]
