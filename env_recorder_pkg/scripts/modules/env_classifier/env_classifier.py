import os
import sys
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix,_classification
from sklearn.multioutput import MultiOutputClassifier
from typing import Any

sys.path.insert(0, '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/scripts/modules')
from bag_parser.bag_parser import Parser, is_bag_dir, is_bag_file

DATASET = {'train':'/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models/train',
            'test':'/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models/test'}

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
        self.clf_pipeline.fit(X.values(),y.values())

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
        

def import_data(data_folder,random_state = 100):
    # iteratung folder files and append to df0
    df = pd.DataFrame()
    for filename in os.scandir(data_folder): 
        if filename.is_file() and filename.path.split('.')[-1]=='csv':
            file_name = filename.path
            df_tmp = pd.read_csv(file_name,index_col=0)
            df = df.append(df_tmp)
    
    df = shuffle(df,random_state=random_state)
    if 'labels' in df.columns:
        X = df.drop(['labels'],axis=1)
        y = df['labels']
    else:
        X = df.drop(['top_labels','mid_labels','bot_labels'],axis=1)
        y = df[['top_labels','mid_labels','bot_labels']]

    return X, y


## need to change prepare data to train
def prepare_dataset(train_folder,test_folder):
    
    X_train,y_train = import_data(train_folder)    
    X_test, y_test = import_data(test_folder)
    
    return X_train.values, X_test.values, y_train.values, y_test.values

def run_train_test():
    
    ### will be changed in refactoring
    X_train, X_test, y_train, y_test = prepare_dataset(data_folder='/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models/test_multi_train')
    mrf = MultiOutputClassifier(RandomForestClassifier()) 
    rf_pipe = Pipeline([('scaler', StandardScaler()), ('model',mrf )])
    rf_env_pipe = EnvClassifierPipe({'clf_pipeline':rf_pipe})
    rf_env_pipe.train(X_train,y_train)
    rf_env_pipe.test(X_test.values(),y_test.values())
    name = input("Press enter = finish without saving pipe | File name = finish with saving pipe \nEnter your input: ")
    if name != "":
        rf_env_pipe.save(name)

def run_models_comparison(multilabel, save=False):
    now = datetime.now()
    if multilabel:
        train_folder = os.path.join(DATASET['train'],'train_multi')
        test_folder = os.path.join(DATASET['test'],'test_multi')
    else: 
        train_folder = os.path.join(DATASET['train'],'train')
        test_folder = os.path.join(DATASET['test'],'test')
    
    X_train, X_test, y_train, y_test = prepare_dataset(train_folder = train_folder, test_folder = test_folder)
    svm_model = SVC()
    if multilabel:
        knn_model = MultiOutputClassifier(KNeighborsClassifier())
        knn_param_grid = {'model__estimator__n_neighbors': [3, 5, 7, 9, 11]}
        rf_model = MultiOutputClassifier(RandomForestClassifier())
        rf_param_grid = {'model__estimator__n_estimators': [10, 50, 100, 200],'model__estimator__max_depth': [None, 5, 10, 15]}

    else:
        knn_model = KNeighborsClassifier()
        knn_param_grid = {'model__n_neighbors': [3, 5, 7, 9, 11]}
        rf_model = RandomForestClassifier()
        rf_param_grid = {'model__n_estimators': [10, 50, 100, 200],'model__max_depth': [None, 5, 10, 15]}
    
    knn_pipe = Pipeline([('scaler', StandardScaler()),('model',knn_model)])
    knn_env_pipe = EnvClassifierPipe({'clf_pipeline':knn_pipe})
    knn_env_pipe.set_grid_search(pipe_params=knn_param_grid,verbose=1)
    knn_env_pipe.pipe_grid.fit(X_train, y_train)
    knn_predictions = knn_env_pipe.pipe_grid.predict(X_test)
    
    print(f"KNN best model score: {knn_env_pipe.pipe_grid.best_score_}")
    print(f"KNN best model score on the test set: {knn_env_pipe.pipe_grid.best_estimator_.score(X_test,y_test)}")
    print(f"KNN best model params: {knn_env_pipe.pipe_grid.best_params_}")
    
    
    rf_pipe = Pipeline([('scaler', StandardScaler()), ('model', rf_model)])
    rf_env_pipe = EnvClassifierPipe({'clf_pipeline':rf_pipe})
    rf_env_pipe.set_grid_search(pipe_params=rf_param_grid,verbose=1)
    rf_env_pipe.pipe_grid.fit(X_train, y_train)
    rf_predictions = rf_env_pipe.pipe_grid.predict(X_test)

    print(f"RF best model score: {rf_env_pipe.pipe_grid.best_score_}")
    print(f"RF best model score on the test set: {rf_env_pipe.pipe_grid.best_estimator_.score(X_test,y_test)}")
    print(f"RF best model params: {rf_env_pipe.pipe_grid.best_params_}")
    

    if not(multilabel):

        svm_pipe = Pipeline([('scaler', StandardScaler()),('model', svm_model)])
        svm_env_pipe = EnvClassifierPipe({'clf_pipeline':svm_pipe})
        svm_param_grid = {'model__C': [0.1, 1, 10, 100],
                        'model__gamma': [1, 0.1, 0.01, 0.001],
                        'model__kernel': ['rbf']}
        svm_env_pipe.set_grid_search(pipe_params=svm_param_grid,verbose=1)
        svm_env_pipe.pipe_grid.fit(X_train, y_train)
        svm_predictions = svm_env_pipe.pipe_grid.predict(X_test)

        print(f"SVM best model score: {svm_env_pipe.pipe_grid.best_score_}")
        print(f"SVM best model score on the test set: {svm_env_pipe.pipe_grid.best_estimator_.score(X_test,y_test)}")
        print(f"SVM best model params: {svm_env_pipe.pipe_grid.best_params_}")
        print("Reports.....")
        print("KNN Classification Report:")
        print(classification_report(y_test, knn_predictions))
        print("RF Classification Report:")
        print(classification_report(y_test, rf_predictions))
        print("SVM Classification Report:")
        print(classification_report(y_test, svm_predictions))

        scores = {'SVM':svm_env_pipe.pipe_grid.best_score_,'KNN':knn_env_pipe.pipe_grid.best_score_,'RF':rf_env_pipe.pipe_grid.best_score_}
        pipes = {'SVM':svm_env_pipe,'KNN':knn_env_pipe,'RF':rf_env_pipe}
    
    else:

        scores = {'KNN':knn_env_pipe.pipe_grid.best_score_,'RF':rf_env_pipe.pipe_grid.best_score_}
        pipes = {'KNN':knn_env_pipe,'RF':rf_env_pipe}
        

    best_score = max(scores.values())
    best_pipe_key = max(scores, key=scores.get)
    best_pipe = pipes[best_pipe_key]
    best_pipe.pipe = best_pipe.pipe_grid.best_estimator_
    
    if not(multilabel):
        best_cm = confusion_matrix(y_test,best_pipe.pipe_grid.predict(X_test),labels= best_pipe.pipe_grid.best_estimator_.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=best_cm,display_labels=best_pipe.pipe_grid.best_estimator_.classes_)
        disp.plot()
        plt.suptitle("Confusion Matrix Of Best Pipe")
        plt.title(f"Best - {best_pipe_key} Pipe")
        plt.show()
    
    if save:
        if multilabel:
            name = "multi_best_"+now.strftime("%d-%m-%Y_%H-%M-%S")
            best_pipe.save(name)
        else:
            name = "best_"+now.strftime("%d-%m-%Y_%H-%M-%S")
            best_pipe.save(name)


class CalssifierParser(Parser):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description ='Bag Iterator')

        parser.add_argument('--cs',action='store_true', help="Compare ML models tests for single label")
        parser.add_argument('--cm',action='store_true', help="Compare ML models tests for multi label")

        return parser.parse_args()


def main():
    
    args = CalssifierParser.get_args()
    if args.cs:
        run_models_comparison(multilabel=False,save=True)

    
    elif args.cm:
        run_models_comparison(multilabel=True,save=True)
    
    else:
        run_train_test()


if __name__ == "__main__":
    main()
