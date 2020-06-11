## Project Details
Below are additional details about each component


## Data Pipelines: Jupyter Notebooks
This is a  folder contains datasets and ipynb files where wrangling process and raw pipelines were developed.

## ETL
[ETL Pipeline Preparation.ipynb](https://github.com/victorlifan/Disaster-Response-Pipelines/blob/master/Jupyter%20Notebook%20file/ETL%20Pipeline%20Preparation.ipynb)

The first part of the data pipeline is the Extract, Transform, and Load process. Here, I preformed read the dataset, clean the data, and then store it in a SQLite database.

The cleaning code is then transferred into [`..IDE/data/process_data.py`](https://github.com/victorlifan/Disaster-Response-Pipelines/tree/master/IDE/data)

## Machine Learning Pipeline
[ML Pipeline Preparation.ipynb](https://github.com/victorlifan/Disaster-Response-Pipelines/blob/master/Jupyter%20Notebook%20file/ML%20Pipeline%20Preparation.ipynb)

For the machine learning portion, I split the data into a training set and a test set. Then, I created a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, I exported the model to a pickle file. After completing the notebook, the final machine learning code is included in [`..IDE/data/train_classifier.py`](https://github.com/victorlifan/Disaster-Response-Pipelines/tree/master/IDE/models)

## Files

* `categories.csv`: id and uncleaned disaster categories
* `messages.csv`: id, translated messages, original messages and message genre
* `InsertDatabaseName.db`: cleaned sql database
* `model.sav`: saved trained ML model
