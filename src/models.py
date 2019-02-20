import pandas as pd
import numpy as np
from os import path
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate

base_dir = 'src/data/'
csv_gv_name = 'vocab.csv'


if __name__ == "__main__":
    data_gv = pd.read_csv(path.join(base_dir, csv_gv_name), sep=",")
    sgv = MultinomialNB(alpha=1.0)
    scores = cross_validate(
        sgv, data_gv.iloc[:, 2:], data_gv['sentiment'], cv=10)
    print(scores)
