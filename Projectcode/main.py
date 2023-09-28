import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from bs4 import BeautifulSoup

 
def get_dataset():
    global data
    data = pd.read_csv('synthetic_water_consumption_data_with_details.csv')
    return "Data is Ready to use"


def train_dataset():
    global fraud_threshold
    global X
    global y
    global X_train
    global y_train
    global y_test
    global scaler
    global X_test
    global knn_classifier
    global svm_classifier
    fraud_threshold = 40  
    data['is_fraud'] = (data['consumption'] < fraud_threshold).astype(int)
    X = data[['consumption']]  
    y = data['is_fraud']  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train, y_train)
    traindata={
        'X_train': X_train,
        'y_train': y_train
    }
    return traindata


def test_dataset_knn():
    global knn_predictions
    knn_predictions = knn_classifier.predict(X_test)
    cm=confusion_matrix(y_test, knn_predictions)
    cr=classification_report(y_test, knn_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix (k-NN)")
    plt.show()
    knn_data={
        'cm':cm,
        'cr':cr
    }
    return knn_data

def test_dataset_svm():
    global svm_predictions
    svm_predictions = svm_classifier.predict(X_test)
    cm=confusion_matrix(y_test, svm_predictions)
    cr=classification_report(y_test, svm_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix (SVM)")
    plt.show()
    svm_data={
        'cm':cm,
        'cr':cr
    }
    return svm_data

def process_test_data():
    global ensemble_predictions
    pt="Processed Test Data"
    ensemble_predictions = (knn_predictions + svm_predictions) >= 1
    cm=confusion_matrix(y_test, ensemble_predictions)
    cr=classification_report(y_test, ensemble_predictions)
    results = {
        'processed_data': pt,
        'confusion_matrix': cm.tolist(),  
        'classification_report': cr
    }
    pt_data={
        'cm':cm,
        'cr':cr
    }
    return pt_data

def display_fraud_users():
    global fraudulent_users
    fd="Fraudulent Users:"
    fraudulent_users = data[data['is_fraud'] == 1]
    return fraudulent_users


def find_user(user_id):
    global user_info
    if (fraudulent_users['user_id'] == int(user_id)).any():
        user_info = data[data['user_id'] == int(user_id)]
        return user_info
    elif (data['user_id'] == int(user_id)).any():
        user_info="User not found in the Fraudulent user's list"
        return user_info
    user_info="Invalid User Id"
    return user_info


def show_accuracy():
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    classifiers = ['KNN', 'SVM']
    accuracies = [knn_accuracy, svm_accuracy]
    plt.bar(classifiers, accuracies, color=['blue', 'green'])
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of KNN vs. SVM')
    plt.ylim(0, 1) 
    plt.show()
    ac={
        'kc':knn_accuracy,
        'sc':svm_accuracy
    }
    return ac
