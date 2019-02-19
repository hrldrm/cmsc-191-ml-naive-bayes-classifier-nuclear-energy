import pandas as pd
import numpy as np
from os import path
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix

base_dir = 'src/data/'
csv_gv_name = 'vocab.csv'


def report_model(model_name, label_test, model_predictions):
    print("Model: {}".format(model_name))
    print(classification_report(label_test, model_predictions))
    print(confusion_matrix(label_test, model_predictions))


if __name__ == "__main__":
    data_gv = pd.read_csv(path.join(base_dir, csv_gv_name), sep=",")
    # sen_train, sen_test, label_train, label_test = train_test_split(data_gv.iloc[:, 2:],
    #                                                                 data_gv['sentiment'],
    #                                                                 test_size=0.2)
    sgv = MultinomialNB(alpha=1.0)
    scores = cross_validate(
        sgv, data_gv.iloc[:, 2:], data_gv['sentiment'], cv=10)
    print(scores)
