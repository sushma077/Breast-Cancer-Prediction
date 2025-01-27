from flask import Flask, render_template, request,  url_for, flash, redirect
import pandas as pd
import numpy as np
import pickle
import os

app=Flask(__name__)


curr=os.path.dirname(__file__)
model_path=os.path.join(curr, "cancer_model_prediction.pkl")

with open(model_path,"rb") as f:
    classifier=pickle.load(f)



@app.route('/')

@app.route('/cancer')
def home():
    return render_template("cancer.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method=='POST':
        ConcavePoints=float(request.form["ConcavePoints"])
        Area=float(request.form["Area"])
        Radius=float(request.form["Radius"])
        Perimeters=float(request.form["Perimeters"])
        Concavity=float(request.form["Concavity"])
        

        input_data=(ConcavePoints,Area, Radius, Perimeters,Concavity)
        input_data_as_numpy_array=np.asarray(input_data)
        input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
        prediction=classifier.predict(input_data_reshaped)

        if(prediction[0]==1):
            result="Sorry, you have chances of getting the disease. Please consult the doctor immediately."
        else:
            result="No need to fear. You have no dangerous symptoms of the disease."

        return render_template("result.html", result=result)
    

if __name__=='__main__':
    app.run(debug=True)
