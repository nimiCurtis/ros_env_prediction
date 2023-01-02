import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import set_config
set_config(display='diagram')
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.multioutput import MultiOutputClassifier
from typing import Any

DATA_FOLDER = '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models/train'

class EnvClassifierPipe:

    def __init__(self,attributes:dict=None):
        self.clf_pipeline = None
        self.pipe_grid = None
        self.models_datafolders = '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models'
        if attributes is not None:
            for key,value in attributes.items():
                setattr(self,key,value)    
    
    @property
    def pipe(self)->Pipeline:
        return self.clf_pipeline

    @pipe.setter
    def pipe(self,pipe:Pipeline):
        self.clf_pipeline = pipe

    def load(self,file_name):
        pipe_file = os.path.join(self.models_datafolders,file_name)
        pipe = load(pipe_file)
        self.clf_pipeline = pipe 

    def save(self,name):
        name = name + '.joblib'
        pipe_file = os.path.join(self.models_datafolders,name)
        dump(self.clf_pipeline,pipe_file)
        print(f"[INFO]  Pipe {pipe_file} saved.")
    
    def fit(self,X,y=None):
        self.clf_pipeline.fit(X,y)

    def predict(self,x):
        predict = self.clf_pipeline.predict(x)
        return predict

    def score(self,X_t,y_t):
        score = self.clf_pipeline.score(X_t,y_t)
        return score
    
    def predict_proba(self,x):
        preds = self.clf_pipeline.predict_proba(x)
        return preds

    def train(self,X,y):
        self.fit(X.values,y.values)

    def test(self,X_t,y_t):
        score = self.score(X_t,y_t)
        print(f"Model accuracy is: {score}")
    
    def set_grid_search(self,pipe_params,verbose=2,report=False):
        self.pipe_grid = GridSearchCV(self.clf_pipeline,pipe_params,verbose=verbose)
        



def prepare_data_to_train(data_folder,random_state = 100,test_size=0.2):
    df = pd.DataFrame()

    # iteratung folder files and append to df0
    for filename in os.scandir(data_folder): 
        if filename.is_file() and filename.path.split('.')[-1]=='csv':
            file_name = filename.path
            df_tmp = pd.read_csv(file_name,index_col=0)
            df = df.append(df_tmp)
    
    if 'labels' in df.columns:
        X = df.drop(['labels'],axis=1)
        y = df['labels']
    else:
        X = df.drop(['top_labels','mid_labels','bot_labels'],axis=1)
        y = df[['top_labels','mid_labels','bot_labels']]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,test_size=test_size)

    return X_train, X_test, y_train, y_test

def run_train_test():
    
    X_train, X_test, y_train, y_test = prepare_data_to_train(data_folder='/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models/test_multi_train')
    mrf = MultiOutputClassifier(RandomForestClassifier()) 
    rf_pipe = Pipeline([('scaler', StandardScaler()), ('model',mrf )])
    rf_env_pipe = EnvClassifierPipe({'clf_pipeline':rf_pipe})
    rf_env_pipe.train(X_train,y_train)
    rf_env_pipe.test(X_test,y_test)
    name = input("Press enter = finish without saving pipe | File name = finish with saving pipe \nEnter your input: ")
    if name != "":
        rf_env_pipe.save(name)

def run_models_comparison(save=False):
    now = datetime.now()
    X_train, X_test, y_train, y_test = prepare_data_to_train(data_folder = DATA_FOLDER)
    
    svm_pipe = Pipeline([('scaler', StandardScaler()),('model', SVC())])
    knn_pipe = Pipeline([('scaler', StandardScaler()),('model',KNeighborsClassifier())])
    rf_pipe = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())])
    
    svm_env_pipe = EnvClassifierPipe({'clf_pipeline':svm_pipe})
    knn_env_pipe = EnvClassifierPipe({'clf_pipeline':knn_pipe})
    rf_env_pipe = EnvClassifierPipe({'clf_pipeline':rf_pipe})

    svm_param_grid = {'model__C': [0.1, 1, 10, 100],
                    'model__gamma': [1, 0.1, 0.01, 0.001],
                    'model__kernel': ['rbf']}
    knn_param_grid = {'model__n_neighbors': [3, 5, 7, 9, 11]}
    rf_param_grid = {'model__n_estimators': [10, 50, 100, 200],'model__max_depth': [None, 5, 10, 15]}

    svm_env_pipe.set_grid_search(pipe_params=svm_param_grid,verbose=1)
    knn_env_pipe.set_grid_search(pipe_params=knn_param_grid,verbose=1)
    rf_env_pipe.set_grid_search(pipe_params=rf_param_grid,verbose=1)


    svm_env_pipe.pipe_grid.fit(X_train, y_train)
    knn_env_pipe.pipe_grid.fit(X_train, y_train)
    rf_env_pipe.pipe_grid.fit(X_train, y_train)

    svm_predictions = svm_env_pipe.pipe_grid.predict(X_test)
    #print(svm_env_pipe.pipe_grid.best_estimator_.score(X_test,y_test))
    knn_predictions = knn_env_pipe.pipe_grid.predict(X_test)
    rf_predictions = rf_env_pipe.pipe_grid.predict(X_test)

    print(f"SVM best model score: {svm_env_pipe.pipe_grid.best_score_}")
    print(f"SVM best model score on the test set: {svm_env_pipe.pipe_grid.best_estimator_.score(X_test,y_test)}")
    print(f"SVM best model params: {svm_env_pipe.pipe_grid.best_params_}")
    print("SVM Classification Report:")
    print(classification_report(y_test, svm_predictions))
    
    
    print(f"KNN best model score: {knn_env_pipe.pipe_grid.best_score_}")
    print(f"KNN best model score on the test set: {knn_env_pipe.pipe_grid.best_estimator_.score(X_test,y_test)}")
    print(f"KNN best model params: {knn_env_pipe.pipe_grid.best_params_}")
    print("KNN Classification Report:")
    print(classification_report(y_test, knn_predictions))
    
    print(f"RF best model score: {rf_env_pipe.pipe_grid.best_score_}")
    print(f"RF best model score on the test set: {rf_env_pipe.pipe_grid.best_estimator_.score(X_test,y_test)}")
    print(f"RF best model params: {rf_env_pipe.pipe_grid.best_params_}")
    print("RF Classification Report:")
    print(classification_report(y_test, rf_predictions))

    scores = {'SVM':svm_env_pipe.pipe_grid.best_score_,'KNN':knn_env_pipe.pipe_grid.best_score_,'RF':rf_env_pipe.pipe_grid.best_score_}
    pipes = {'SVM':svm_env_pipe,'KNN':knn_env_pipe,'RF':rf_env_pipe}
    best_score = max(scores.values())
    best_pipe_key = max(scores, key=scores.get)
    best_pipe = pipes[best_pipe_key]
    best_pipe.pipe = best_pipe.pipe_grid.best_estimator_
    
    best_cm = confusion_matrix(y_test, svm_predictions, labels= best_pipe.pipe_grid.best_estimator_.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=best_cm,display_labels=best_pipe.pipe_grid.best_estimator_.classes_)
    disp.plot()
    plt.suptitle("Confusion Matrix Of Best Pipe")
    plt.title(f"Best - {best_pipe_key} Pipe")
    plt.show()
    
    if save:
        name = "best_"+now.strftime("%d-%m-%Y_%H-%M-%S")
        best_pipe.save(name)
    

def main():
    run_train_test()
    #run_models_comparison(save=True)
    

if __name__ == "__main__":
    main()
