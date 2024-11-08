import os
import pickle
from flask import Flask, render_template, request
import yaml

with open("./config/config.yaml","r") as file:
    config =  yaml.safe_load(file)

model_path = config["flask_app"]["best_model_path"]
model_path_filename = os.path.join(model_path,"linear_reg","model.pkl")
model = pickle.load(open(model_path_filename,"rb"))

app=Flask(__name__)

@app.route("/", methods=["GET","POST"])
def prediction():
    if request.method == "POST":
        data_dict = dict(request.form)
        area = float(data_dict["area"])
        prediction = model.predict([[area]])[0][0]
        return render_template("test.html",prediction=prediction)
    else:
        return render_template("test.html")
    
if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)