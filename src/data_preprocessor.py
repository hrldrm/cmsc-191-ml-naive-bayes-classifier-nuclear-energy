import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
from datetime import datetime
from os import path

dataset_path = "src/data/"
dataset_name = "data.csv"
csv_path = 'processed-{}.csv'.format(
    datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))


def preprocess(index):
    # had to put `engine=python` because of an encoding error
    data = pd.read_csv(index, sep=",", engine='python', dtype="object")
    data = data.drop(['sentiment_confidence_summary'], axis=1)
    # sentiments = pd.DataFrame(data.groupby('sentiment').agg(
    # {'sentiment': ['count']}))
    data['sentiment'] = data['sentiment'].map(
        {'Negative': 0, 'Positive': 1, 'Neutral / author is just sharing information': 2, 'Tweet NOT related to nuclear energy': 3})
    data['tweet_text_tokens'] = data['tweet_text'].apply(_preprocess_text)
    data = data.dropna()
    data.to_csv(path.join(dataset_path, csv_path))
    print(data)


def _preprocess_text(text):
    text = re.sub('RT|@mention|{.*}|\[.*\]|\&.*\;|�_�|�|��|[0-9]', '', text)
    text = [char for char in text
            if char.lower() not in string.punctuation]
    text = ''.join(text)
    return [word for word in text.split() if word.lower() not in
            stopwords.words('english')]


if __name__ == "__main__":
    preprocess(path.join(dataset_path, dataset_name))
