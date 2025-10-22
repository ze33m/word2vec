from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
import re
from debug_dataset import debug_dataset
import pickle
from datasets import load_dataset
import yaml
import datetime
from multiprocessing import Pool, cpu_count
stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer("russian")


def now():
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')

def preprocess_docs(raw_docs):
    n_jobs = max(1, cpu_count() - 1)
    with Pool(processes=n_jobs) as pool:
        docs = list(tqdm(pool.imap(preprocessing, raw_docs), total=len(raw_docs)))

    return docs

def preprocessing(raw_text:str):
    raw_text = raw_text.lower()
    raw_text = re.sub(r'[^\w\s]|\d', '', raw_text) 
    raw_text = raw_text.replace('\n', ' ')
    raw_text = re.sub(r'[\s]+', ' ', raw_text)
    tokens = [stemmer.stem(i) for i in raw_text.split() if i not in stop_words]
    return tokens

if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config['dataset']['DEBUG']:   
        dataset = debug_dataset
    else:
        dataset = load_dataset('0x7o/taiga', split='train')
        dataset = [item['text'] for item in tqdm(dataset, desc='Извлечение текстов')]

    docs = preprocess_docs(dataset)

    with open(f'datasets/{now()}.pkl', 'wb') as file:
            pickle.dump(docs, file)
