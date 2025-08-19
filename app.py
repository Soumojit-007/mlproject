import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline 
import socket

application = Flask(__name__)
app = application


## Route for home page
@app.route('/')
def index():
    return render_template('index.html')
def find_free_port():
    s = socket.socket()
    s.bind(('' , 0))
    port = s.getsockname()[1]
    s.close()
    return port

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
    gender=request.form.get('gender', 'female'),  # default value
    race_ethnicity=request.form.get('race/ethnicity', 'group B'),
    parental_level_of_education=request.form.get('parental_level_of_education', 'some college'),
    lunch=request.form.get('lunch', 'standard'),
    test_preparation_course=request.form.get('test_preparation_course', 'none'),
    reading_score=float(request.form.get('reading_score', 0)),
    writing_score=float(request.form.get('writing_score', 0))
)


        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    free_port = find_free_port()
    app.run(host="0.0.0.0", port=free_port, debug=True, threaded=True)
