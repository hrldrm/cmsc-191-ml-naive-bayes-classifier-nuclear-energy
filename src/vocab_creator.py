import os
import pandas as pd
import numpy as np

from ast import literal_eval

dataset_path = "src/data/"
dataset_name = "processed-2019-02-18-20-47-04.csv"
vocab_name = 'vocab.csv'


def get_word_counts(text_df):
    word_count_df = pd.DataFrame(columns=['sentiment'])
    for index, row in text_df.iterrows():
        tokens = [word.lower() for word in row['tweet_text_tokens']]
        for token in tokens:
            if token not in word_count_df.columns:
                word_count_df[token] = ''

        curr_tokens = {feat.lower(): tokens.count(feat.lower())
                       for feat in list(word_count_df)}
        curr_tokens['sentiment'] = row['sentiment']
        word_count_df = word_count_df.append(curr_tokens, ignore_index=True)

    return word_count_df.replace('', 0)


if __name__ == '__main__':
    nuclear_senti_df = pd.read_csv(os.path.join(dataset_path, dataset_name),
                                   usecols=['sentiment', 'tweet_text_tokens'],
                                   converters={'tweet_text_tokens': literal_eval})

    vocab_df = get_word_counts(nuclear_senti_df)
    print(vocab_df)
    vocab_df.to_csv(os.path.join(dataset_path, vocab_name))
