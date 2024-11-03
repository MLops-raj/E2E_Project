import os
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import yaml
import argparse

with open("./../config/config.yaml","r") as file:
    config =  yaml.safe_load(file)

processed_data_path = config['data_paths']['processed_data_path']
model_path = config['data_paths']['model_path']


def model_eval(processed_data_path,model_path):
    print("#################Model Evaluation Started################")
    model_path_filename = os.path.join(model_path,"linear_model.pkl")
    model = pickle.load(open(model_path_filename,'rb'))

    X_test_path = os.path.join(processed_data_path,"X_test.csv")
    Y_test_path = os.path.join(processed_data_path,"y_test.csv")

    x_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(Y_test_path)

    y_pred_test = model.predict(x_test)

    r2_score_test = r2_score(y_test,y_pred_test)
    MAE_test = mean_absolute_error(y_test,y_pred_test)
    MSE_test = mean_squared_error(y_test,y_pred_test)

    mlflow.log_metrics({"r2_score_test":r2_score_test,"MAE_test":MAE_test,"MSE_test":MSE_test})

    x_train_path = os.path.join(processed_data_path,"X_train.csv")
    y_train_path = os.path.join(processed_data_path,"y_train.csv")

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    y_pred_train = model.predict(x_train)

    r2_score_train = r2_score(y_train,y_pred_train)
    MAE_train = mean_absolute_error(y_train,y_pred_train)
    MSE_train = mean_squared_error(y_train,y_pred_train)
    mlflow.log_metrics({"r2_score_train":r2_score_train,"MAE_train":MAE_train,"MSE_train":MSE_train})
    mlflow.sklearn.log_model(model,"linear_reg")
    print("#################Model Evaluation Finished ################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_path",help="provide raw data path", default=processed_data_path)
    parser.add_argument("--model_path",help="provide cleaned data path", default=model_path)
    args = parser.parse_args()
    model_eval(args.processed_data_path,args.model_path)