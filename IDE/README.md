## Project Details
Below are additional details about each component.


### File structure
![Alt text](https://github.com/victorlifan/Disaster-Response-Pipelines/blob/master/screenshots/structure.png?raw=true)

### Python Scripts:

After I completed the notebooks for the ETL and machine learning pipeline, I transferred codes into Python scripts, `process_data.py` and `train_classifier.py`. These Python scripts is able to run with additional arguments specifying the files used for the data and model.

### Running the Web App

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/
