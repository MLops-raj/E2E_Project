import pandas as pd 
import os
import argparse
import yaml
import mlflow
from sklearn.model_selection import train_test_split

with open("./../config/config.yaml","r") as file:
    config =  yaml.safe_load(file)

test_size = config['preprocess']['test_size']
Target = config['preprocess']['Target']
cleaned_data_path = config['data_paths']['cleaned_data_path']
processed_data_path = config['data_paths']['processed_data_path']


def processed_data(cleaned_data_path, processed_data_path, Target):
    cleaned_data_file = os.listdir(cleaned_data_path)[0]
    print(cleaned_data_file)
    print("################DATA prepprocessing STARTED##################")
    cleaned_data = os.path.join(cleaned_data_path,cleaned_data_file)
    print(cleaned_data)
    df = pd.read_csv(cleaned_data)
    X = df.drop(columns=[Target])
    Y = df[[Target]]
    print(X)
    print(Y)

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=test_size)

    mlflow.log_param("test_size",test_size)

    X_train.to_csv(os.path.join(processed_data_path,"X_train.csv"),index=False)
    X_test.to_csv(os.path.join(processed_data_path,"X_test.csv"),index=False)
    y_train.to_csv(os.path.join(processed_data_path,"y_train.csv"),index=False)
    y_test.to_csv(os.path.join(processed_data_path,"y_test.csv"),index=False)

    print("################DATA PREPROCESSING FINISHED#####################")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_data_path",help="provide cleaned data path", default=cleaned_data_path)
    parser.add_argument("--processed_data_path",help="provide cleaned data path", default=processed_data_path)
    parser.add_argument("--Target",help="provide cleaned data path", default=Target)

    args = parser.parse_args()
    processed_data(args.cleaned_data_path,args.processed_data_path,args.Target)