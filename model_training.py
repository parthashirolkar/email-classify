import os
from typing import Tuple
import re
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandarallel import pandarallel 

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def clean_text(text: str) -> str:
    """
    Cleans and preprocesses a text string by removing non-alphabetic characters, converting to lowercase,
    removing stopwords, and applying stemming.

    Parameters:
    text (str): The input text string to be cleaned.

    Returns:
    str: The cleaned and preprocessed text string.

    Raises:
    TypeError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
    
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [stemmer.stem(word.lower()) for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

def save_metrics(cv: CountVectorizer, model: MultinomialNB):
    """
    Creates a DataFrame of features and their corresponding probabilities for each class
    from a trained CountVectorizer and MultinomialNB model.

    Parameters:
    cv (CountVectorizer): A fitted CountVectorizer instance used to transform the text data.
    model (MultinomialNB): A trained Multinomial Naive Bayes model.

    Returns:
    pd.DataFrame: A DataFrame containing features and their class probabilities.

    Raises:
    TypeError: If the inputs are not of the expected types.
    """
    if not isinstance(cv, CountVectorizer):
        raise TypeError("cv must be an instance of sklearn.feature_extraction.text.CountVectorizer.")
    if not isinstance(model, MultinomialNB):
        raise TypeError("model must be an instance of sklearn.naive_bayes.MultinomialNB.")
    if not hasattr(model, 'feature_log_prob_'):
        raise ValueError("The model does not have feature_log_prob_ attribute. Ensure the model is trained.")
    
    metric_df = pd.DataFrame()
    metric_df['feature'] = cv.get_feature_names_out()
    metric_df['class_0_probs'] =  np.exp(1)**model.feature_log_prob_[0]
    metric_df['class_1_probs'] =  np.exp(1)**model.feature_log_prob_[1]
    return metric_df

def initialize_parallel():
    """
    Initializes parallel processing with pandarallel using the number of available CPU cores.
    """
    pandarallel.initialize(nb_workers=os.cpu_count(), progress_bar=True)

def preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocesses the input data by reading a CSV file, cleaning text, and splitting into train and test sets.

    Parameters:
    file_path (str): Path to the input CSV file containing text and labels.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test split for training and testing.
    """
    df = pd.read_csv(file_path)
    df['text'] = df['text'].parallel_apply(clean_text)
    df = df.dropna()
    X = df['text']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def vectorize_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Vectorizes text data using CountVectorizer with specific parameters.

    Parameters:
    X_train (pd.DataFrame): Training text data.
    X_test (pd.DataFrame): Testing text data.

    Returns:
    Tuple[CountVectorizer, pd.DataFrame, pd.DataFrame]:
        The vectorizer instance, transformed training data, and transformed testing data.
    """
    vectorizer = CountVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2), stop_words='english', lowercase=True)
    return vectorizer, vectorizer.fit_transform(X_train), vectorizer.transform(X_test)

def train_model(X_train_vectorized, y_train) -> MultinomialNB:
    """
    Trains a Multinomial Naive Bayes model using GridSearchCV to optimize hyperparameters.

    Parameters:
    X_train_vectorized: Vectorized training data.
    y_train: Training labels.

    Returns:
    MultinomialNB: The trained model.
    """
    params = {'alpha': np.arange(0.01, 2.0, 0.1, dtype=np.float32)}
    grid = GridSearchCV(MultinomialNB(), params, cv=5, n_jobs=-1, verbose=1, scoring='f1_macro')
    grid.fit(X_train_vectorized, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best score: {grid.best_score_}")
    model = MultinomialNB(**grid.best_params_)
    model.fit(X_train_vectorized, y_train)
    return model

def evaluate_and_save(model: MultinomialNB, X_test_vectorized, y_test, vectorizer):
    """
    Evaluates the model, saves the confusion matrix as an image, and stores the metrics as a CSV file.

    Parameters:
    model: The trained Multinomial Naive Bayes model.
    X_test_vectorized: Vectorized testing data.
    y_test: Testing labels.
    vectorizer: The CountVectorizer instance used for feature extraction.
    """
    preds = model.predict(X_test_vectorized)
    print(metrics.classification_report(y_test, preds))
    cm = metrics.confusion_matrix(y_test, preds)
    fig = metrics.ConfusionMatrixDisplay.from_estimator(model, X_test_vectorized, y_test)
    fig.plot()
    plt.savefig('confusion_matrix.png')
    metrics_df = save_metrics(vectorizer, model)
    metrics_df.sort_values(['class_0_probs', 'class_1_probs'], ascending=False).to_csv('csv/metrics.csv', index=False)

def save_artifacts(model, vectorizer):
    """
    Saves the trained model and vectorizer as pickle files.

    Parameters:
    model: The trained Multinomial Naive Bayes model.
    vectorizer: The CountVectorizer instance used for feature extraction.
    """
    with open("obj/model.pkl", "wb") as f:
        joblib.dump(model, f)
    with open("obj/vectorizer.pkl", "wb") as f:
        joblib.dump(vectorizer, f)

def main():
    initialize_parallel()
    X_train, X_test, y_train, y_test = preprocess_data("csv/combined_data.csv")
    vectorizer, X_train_vectorized, X_test_vectorized = vectorize_data(X_train, X_test)
    model = train_model(X_train_vectorized, y_train)
    evaluate_and_save(model, X_test_vectorized, y_test, vectorizer)
    save_artifacts(model, vectorizer)

if __name__ == "__main__":
    main()