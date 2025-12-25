from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
from datasets import load_dataset
import yaml
from multiprocessing import cpu_count

stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer("russian")

def preprocessing(raw_text:str):
    """
    Функция для предобработки одной строчки
    """
    raw_text = raw_text.lower()
    raw_text = re.sub(r'[^\w\s]|\d', '', raw_text) 
    raw_text = raw_text.replace('\n', ' ')
    raw_text = re.sub(r'[\s]+', ' ', raw_text)
    tokens = [stemmer.stem(i) for i in raw_text.split() if i not in stop_words]
    return tokens

def preprocess_docs(batch):
    """
    Функция для предобработки всех строчек
    """
    texts = batch["text"]
    processed = [preprocessing(i) for i in texts]
    return {"tokens" : processed}

if __name__ == '__main__':

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset = load_dataset('0x7o/taiga', split='train')

    if config['dataset']['DEBUG']:   
        dataset = dataset.select(range(10))

    print(type(dataset))

    # применение preprocess_docs к нашему датасету с распараллеливанием
    dataset = dataset.map(
        preprocess_docs,
        batched=True,
        num_proc=max(1, cpu_count() - 1),
        remove_columns=["text", "source"],
        desc="Preprocessing"
    )

    print(dataset)
    dataset.save_to_disk('preprocessed')


