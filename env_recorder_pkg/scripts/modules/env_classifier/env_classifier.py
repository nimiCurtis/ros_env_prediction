import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import set_config
from joblib import dump, load

class EnvClassifierPipe:

    def __init__(self):
        
        self.models_datafolders = '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models'
        clf = SVC(kernel="rbf",gamma=0.005, C=1000)
        self.clf_pipline = Pipeline(steps=[('scale',StandardScaler()),('model',clf)])

    def load(self,file_name):
        pipe_file = os.path.join(self.models_datafolders,file_name)
        self.clf_pipline = load(pipe_file)

    def save(self,name):
        name = name + '.joblib'
        pipe_file = os.path.join(self.models_datafolders,name)
        dump(self.clf_pipline,pipe_file)
    
    def prepare_data_to_train(self,data_folder,random_state = 100,test_size=0.2):
        df = pd.DataFrame()

        # iteratung folder files and append to df0
        for filename in os.scandir(data_folder): 
            if filename.is_file() and filename.path.split('.')[-1]=='csv':
                file_name = filename.path
                df_tmp = pd.read_csv(file_name,index_col=0)
                df = df.append(df_tmp)

        X = df.drop(['labels'],axis=1)
        y = df['labels']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,test_size=test_size)

        return X_train, X_test, y_train, y_test

    def fit(self,X,y=None):
        self.clf_pipline.fit(X,y)

    def predict(self,x):
        predict = self.clf_pipline.predict(x)
        return predict

    def score(self,X_t,y_t):
        score = self.clf_pipline.score(X_t,y_t)
        return score
    
    def predict_proba(self,x):
        preds = self.clf_pipline.predict_proba(x)
        return preds

    def train(self,X,y):
        self.fit(X.values,y.values)

    def test(self,X_t,y_t):
        score = self.score(X_t,y_t)
        print(f"Model accuracy is: {score}")

    def display(self,disp_type='diagram'):
        set_config(display=disp_type)
        return self.clf_pipeline

def run_train_test(data_folder):
    pipe = EnvClassifierPipe()
    X_train, X_test, y_train, y_test = pipe.prepare_data_to_train(data_folder)
    pipe.train(X_train,y_train)
    pipe.test(X_test,y_test)
    name = input("Press enter = finish without saving pipe | File name = finish with saving pipe \nEnter your input: ")
    if name != "":
        pipe.save(name)

def main():
    run_train_test('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models/train')

if __name__ == "__main__":
    main()
