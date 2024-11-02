import pandas as pd 
import os
import argparse
import yaml
import mlflow

with open("./../config/config.yaml","r") as file:
    config =  yaml.safe_load(file)

raw_data_path = config['data_paths']['raw_data_path']
cleaned_data_path = config['data_paths']['cleaned_data_path']
raw_data_file = config['data_paths']['raw_data_file']

def data_cleaning(raw_data_path,cleaned_data_path,raw_data_file):
    raw_data_file = os.listdir(raw_data_path)[0]
    print("################DATA CLEANING STARTED##################")
    print(raw_data_file)
    raw_data = os.path.join(raw_data_path,raw_data_file)
    df = pd.read_csv(raw_data)

    cleaned_data_file = os.path.join(cleaned_data_path,raw_data_file)
    df.to_csv(cleaned_data_file,index=False)
    mlflow.log_param("cleaned_data_path",clean_data_file)
    print("################DATA CLEANING FINISHED##################")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path",help="provide raw data path", default=raw_data_path)
    parser.add_argument("--cleaned_data_path",help="provide cleaned data path", default=cleaned_data_path)
    parser.add_argument("--raw_data_file",help="provide raw data path", default=raw_data_file)
    args = parser.parse_args()
    data_cleaning(args.raw_data_path,args.cleaned_data_path,args.raw_data_file)

    #if __name__ == "__main__":
    #    parser = argparse.ArgumentParser()
    #    parser.add_argument("--raw_data_path",help="provide raw data path", default=raw_data_path)
    #    parser.add_argument("--cleaned_data_path",help="provide cleaned data path", default=cleaned_data_path)
    #    parser.add_argument("--raw_data_file",help="provide raw data file", default=raw_data_file)
    #    args = parser.parse_args()
    #    data_cleaning(args.raw_data_path,args.cleaned_data_path,args.raw_data_file)