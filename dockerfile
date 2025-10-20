FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords')"

COPY . . 

RUN mkdir -p models

CMD ["python", "prep.py"]