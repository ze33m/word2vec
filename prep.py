from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
import re
from debug_dataset import debug_dataset
import pickle
from datasets import load_dataset
import yaml
import time
from multiprocessing import Pool, cpu_count
stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer("russian")

def preprocess_docs(raw_docs):
    docs = []
    n_jobs = max(1, cpu_count() - 1)
    start = time.perf_counter()
    docs1 = [preprocessing(i) for i in docs]
    end = time.perf_counter()
    print(end-start)
    # print(f'Предобработка корпуса ({n_jobs} процессов)')
    start = time.perf_counter()
    with Pool(processes=n_jobs) as pool:
        docs2 = pool.map(preprocessing, raw_docs)
    end = time.perf_counter()
    print(end-start)
    print(docs1 == docs2)
    return docs

def preprocessing(raw_text:str):
    raw_text = raw_text.lower()
    raw_text = re.sub(r'[^\w\s]|\d', '', raw_text) 
    raw_text = raw_text.replace('\n', ' ')
    raw_text = re.sub(r'[\s]+', ' ', raw_text)
    tokens = [stemmer.stem(i) for i in raw_text.split() if i not in stop_words]
    return tokens

# разобраться с параллельностью

if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config['dataset']['DEBUG']:   
        dataset = debug_dataset
    else:
        dataset = load_dataset('0x7o/taiga', split='train')['text']

    docs = preprocess_docs(dataset)

    with open(f'dataset.pkl', 'wb') as file:
            pickle.dump(docs, file)
