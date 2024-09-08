import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'maxent_ne_chunker', 'words'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
    Load data from an SQLite database and split it into features and target variables.

    input:
    database_filepath: Path to the SQLite database file.

    output:
    X: Series containing the message texts.
    Y: DataFrame containing the target variables.
    category_names: List of category names corresponding to the target variables.
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("InsertTableName", engine)
    # Split X and Y
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and normalize text, removing stopwords and performing lemmatization.

    input:
    text: Text to be tokenized and normalized.

    output:
    clean_tokens: List of cleaned and lemmatized tokens.
    """
    # Clean and Normalization text
    text_clean = re.sub(r'[^\w]', ' ', text.lower())
    # Tokenization text
    tokens = word_tokenize(text_clean)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens=[]
    stop_words = stopwords.words("english")
    for tok in tokens:
        if tok not in stop_words:
            tok_clean = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(tok_clean)

    return clean_tokens


def build_model():
    """
    Build a machine learning model pipeline with GridSearchCV for hyperparameter tuning.

    output:
    cv: GridSearchCV object containing the pipeline and hyperparameters.
    """
    # Initialize model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Using GridSearchCV to find optimiza hyperparameter
    parameters = {
        'clf__estimator__n_estimators': [20, 30],
        # 'clf__estimator__min_samples_split': [2, 3]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the trained model and print classification reports for each category.

    input:
    model: Trained model object.
    X_test: Test features.
    Y_test: True target values for test set.
    category_names: List of category names corresponding to the target variables.
    """
    Y_pred = model.predict(X_test)

    for idx, col in enumerate(category_names):
        print("########################")
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, idx]))


def save_model(model, model_filepath):
    """
    Save the trained model to a file.

    input:
    model: Trained model object.
    model_filepath: Path to the file where the model will be saved.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main function to load data, build and train a model, evaluate it, and save the trained model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()