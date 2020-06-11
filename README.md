## Project Title
Disaster Response Pipelines (ETL, NLP, ML pipeline project)

## by Fan Li

## Date created
Project is created on Jun 11 2020.

## Table of Contents
1. [Description](#description)
2. [Workflow](#Workflow)
	1. [ETL Pipeline](#ETL)
	2. [ML Pipeline](#ML)
	3. [Flask Web App](#Flask)
3. [Dataset](#Dataset)
4. [Summary of Findings](#summary)
5. [About](#About)
6. [Software used](#Software)


<a name="description"></a>
## Description
In this project, I applied ETL,NLP, ML skills to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

In the [Jupyter Notebook file](https://github.com/victorlifan/Disaster-Response-Pipelines/tree/master/Jupyter%20Notebook%20file), you'll find a data set containing real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that a messages can be sent to an appropriate disaster relief agency.

This project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

Below are a few screenshots of the web app.

<img src='screenshots/overall web look.png' width=500px>
<img src='screenshots/msg display.png' width=500px>


<a name="Workflow"></a>
## Workflow:

<a name="ETL"></a>
##### 1. ETL Pipeline
A data clean pipeline that:
+ Loads the `messages` and `categories` datasets
+ Merges the two datasets
+ Cleans the data
+ Stores it in a SQLite database

<a name="ML"></a>
##### 2. ML Pipeline
A machine learning pipeline that:
+ Loads data from the SQLite database
+ Splits the dataset into training and test sets
+ Builds a text processing and machine learning pipeline
+ Trains and tunes a model using GridSearchCV
+ Outputs results on the test set
+ Exports the final model as a pickle file

<a name="Flask"></a>
##### 3. Flask Web App
A Flask web app.
+ Modify file paths for database and model as needed
+ Add data visualizations using Plotly in the web app.


<a name="Dataset"></a>
## Dataset

* `categories.csv`: id and uncleaned disaster categories
* `messages.csv`: id, translated messages, original messages and message genre

<a name="summary"></a>
## Summary of Findings

1. From the 'DIstribution of Message Genres' we can tell `news` has the most records.
2. From the 'Distribution of Disaster Categories' we can tell `aid_related` has the most records
3. The web app can display a multi-classification outcome based on the trained ML model by given any text message. The outcome suggests what are the disaster categories the text might be indicating.

<a name="About"></a>
## About
+ [`Jupyter Notebook file`](https://github.com/victorlifan/Disaster-Response-Pipelines/tree/master/Jupyter%20Notebook%20file): a folder contains datasets and ipynb files where wrangling process and raw pipelines were developed.
+ [`IDE`](https://github.com/victorlifan/Disaster-Response-Pipelines/tree/master/IDE): a folder contains datasets and modularized py files to support flask web app
+ [`screenshots`](https://github.com/victorlifan/Disaster-Response-Pipelines/tree/master/screenshots): png files were displayed in READMEs

<a name="Software"></a>
## Software used
+ Jupyter Notebook
+ Atom
+ Python 3.7
> + Sklearn
> + Numpy
> + pandas
> + re
> + pickle
> + sqlalchemy
> + nltk
> + plotly
> + flask



## Credits
+ Data provided by: [Figure Eight](https://www.figure-eight.com/) through [DATA SCIENTIST NANODEGREE PROGRAM](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
+ Instruction and assist: [DATA SCIENTIST NANODEGREE PROGRAM](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
