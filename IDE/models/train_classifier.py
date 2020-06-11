import sys
import re
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk import word_tokenize,sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer

def load_data(database_filepath):
    '''
    load data from database
    input:
    - database filepath
    output:
    - X text column
    - Y target classification
    - category_names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('InsertTableName',engine)
    X = df['message']
    Y = df.drop('message',axis=1)
    return X, Y, list(Y)

def tokenize(text):
    '''
    a tokenization function to process text data as a part of ML pipeline
    input:
    - text file
    output:
    - a post-processed list of words
    '''
    # exclude stop words and punctuations
    stop_word=stopwords.words('english')
    text=re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ',text)
    text=re.sub(r'[^0-9A-Za-z]',' ',text)
    # tokenize and initialize lemmatizer
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()

    # normalize and lemmatize
    clean=[]
    for i in tokens:
        if i not in stop_word:
            clean_token=lemmatizer.lemmatize(i).lower().strip()
            # lemmatize verbs
            clean_token=lemmatizer.lemmatize(clean_token,pos='v')
            clean.append(clean_token)
    return clean

# add StartingVerbExtractor transformer
def StartingVerbExtractor(texts):
    '''
    a customize function to determine if the text starts with verbs.
    it will be a part of the pipeline
    input:
    - text file
    output:
    - a df of converted text use binary values to indicate if it starts with a verb
    '''
    # convert tests into series first
    texts=pd.Series(texts)
    X_list=[]
    for index, text in texts.items():

        # tokenzie each sentence into words and tag part of speech
        pos_tags=pos_tag(word_tokenize(text))

        # index pos_tags to get the first word and pos
        try:
            first_word,first_tag=pos_tags[0]
            pass

            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB','VBP'] or first_word=='RT':
                X_list.append(1)
            else:
                X_list.append(0)
        # there are two records message are empty, so we append 0
        except IndexError:
            #print('index:',index,'text:', text)
            X_list.append(0)
    return pd.DataFrame(X_list)

def build_model():
    '''
    function to bulid a ml Pipeline
    no input
    output:
    - model classifier
    '''
    pipeline2=Pipeline([
    ('features',FeatureUnion([
        ('tfidfvac',TfidfVectorizer(tokenizer=tokenize)),
        ('starting_verb',FunctionTransformer(StartingVerbExtractor))
    ])),
    ('multiclfada',MultiOutputClassifier(AdaBoostClassifier()))
    ])
    return pipeline2

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    function to report  f1 score, precision and recall for each output category of the dataset
    input:
    - model trained classifier
    - X_test
    - Y_test
    - category_names
    output:
    - detailed evaluation of each categories
    '''
    Y_pre=model.predict(X_test)
    Y_pre=pd.DataFrame(Y_pre)
    print(classification_report(Y_test,Y_pre,target_names=category_names))

def save_model(model, model_filepath):
    '''
    function to save the finalized model
    input:
    - model trained classifier
    - model_filepath
    '''
    pickle.dump(model,open(model_filepath,'wb'))


def main():
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
