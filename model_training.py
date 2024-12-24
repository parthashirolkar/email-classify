import os
import re
import joblib
import matplotlib.pyplot as plt
from pandarallel import pandarallel 
import numpy as np
from scipy import sparse
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    # filtered_words = [word for word in filtered_words if len(word) > 2]
    return ' '.join(filtered_words)

def save_metrics(cv: CountVectorizer, model: MultinomialNB):
    metric_df = pd.DataFrame()
    metric_df['feature'] = cv.get_feature_names_out()
    metric_df['class_0_probs'] =  np.exp(1)**model.feature_log_prob_[0]
    metric_df['class_1_probs'] =  np.exp(1)**model.feature_log_prob_[1]
    return metric_df

def main():
    pandarallel.initialize(nb_workers=os.cpu_count(),progress_bar=True)

    df = pd.read_csv("csv/combined_data.csv")
    df['text'] = df['text'].parallel_apply(clean_text)
    df = df.dropna()

    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create count vectorizer object with valid min_df max_df and ngram_range parameters
    vectorizer = CountVectorizer(min_df=2, max_df=0.8, ngram_range=(1, 2), stop_words='english', lowercase=True)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Create a params grid for MultinomialNB, then select the best parameters using GridSearchCV
    params = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
    grid = GridSearchCV(MultinomialNB(), params, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train_vectorized, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best score: {grid.best_score_}")

    model = MultinomialNB(**grid.best_params_)
    # Fit the model on the whole dataset
    model.fit(sparse.vstack([X_train_vectorized, X_test_vectorized]), y)


    preds = model.predict(X_test_vectorized)
    print(metrics.classification_report(y_test, preds))
    cm = metrics.confusion_matrix(y_test, preds)
    fig = metrics.ConfusionMatrixDisplay(cm)
    fig.plot()
    # plt.savefig('confusion_matrix.png')
    metrics_df = save_metrics(vectorizer, model)
    metrics_df.sort_values(['class_0_probs', 'class_1_probs'], ascending=False).to_csv('metrics.csv', index=False)

    with open("obj/model.pkl", "wb") as f:
        joblib.dump(model, f)

    with open("obj/vectorizer.pkl", "wb") as f:
        joblib.dump(vectorizer, f)

if __name__ == "__main__":
    main()
