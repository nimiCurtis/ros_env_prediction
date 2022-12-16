import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
	

def main():
    
    df1 = pd.read_csv('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-12-12-15-21-45/plots/feature/features.csv',index_col=0)
    df2 = pd.read_csv('/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/bag/2022-12-12-15-23-00/plots/feature/features.csv',index_col=0)
    df = df1.append(df2)

    X = df.drop(['labels'],axis=1)
    #X = df.drop(['labels','ug_mean','ug_std','ud_std'],axis=1)
    y = df['labels']


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100,test_size=0.2)

    numerical_cols = X.columns.to_list()
    # Create a transformer object
    scaler = StandardScaler()
    column_transformer = make_column_transformer((StandardScaler(),numerical_cols))

    X_train = scaler.fit_transform(X_train.values)
#    X_train = column_transformer.fit_transform(X_train)
    X_train = pd.DataFrame(data=X_train, columns=scaler.get_feature_names_out())


    # Building and fit the classifier
    #clf = SVC(kernel='poly',degree=3, gamma=0.01, C=1000)
    clf = SVC(kernel="rbf",gamma=0.005, C=1000)
    clf.fit(X_train.values, y_train.values)

    # Transform the training data
    X_test = scaler.transform(X_test)
    
    #X_test = column_transformer.transform(X_test)
    X_test = pd.DataFrame(data=X_test, columns=scaler.get_feature_names_out())

    # Make predictions and check the accuracy
    predictions = clf.predict(X_test.values)
    print(accuracy_score(y_test, predictions))
    
    dump(clf,'/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models/rbf.joblib')
    dump(scaler, '/home/nimibot/catkin_ws/src/ros_env_prediction/env_recorder_pkg/models/transformer.joblib')

if __name__ == "__main__":
    main()