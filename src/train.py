import pickle
import re
import sys

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def filter_comment(d):
    vocab=[]
    for i in range(d.shape[0]):
        comment=re.sub('[^a-zA-Z]',' ',d['comment_text'][i])
        comment=comment.lower().split()
        comment=[ps.stem(word) for word in comment if word not in s]
        comment=' '.join(comment)
        vocab.append(comment)
    return vocab

if __name__ == "__main__":
    # # loading the dataset
    train = pd.read_csv("../input/train.csv")
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    ps = SnowballStemmer('english')
    s = set(stopwords.words('english'))
    train_data = filter_comment(train)

    tfidf_vector = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 2),
        max_features=10000)

    train_word_features = tfidf_vector.fit_transform(train_data)
    model = LogisticRegression(C=3, solver='liblinear')

    scores = []
    for class_name in classes:
        print("Train class: ", class_name)
        train_target = train[class_name]
        cv_score = np.mean(cross_val_score(model, train_word_features, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(class_name, cv_score))
        model.fit(train_word_features, train_target)
    print('Total average score is {}'.format(np.mean(scores)))

    # Saving model to disk
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(tfidf_vector, open('tfidf_vector.pkl', 'wb'))
