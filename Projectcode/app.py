from flask import Flask, render_template, request, jsonify
from main import get_dataset, train_dataset, test_dataset_knn, test_dataset_svm, process_test_data, show_accuracy, display_fraud_users, find_user

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/top')
def top():
    return render_template('top.html')

@app.route('/bottom')
def bottom():
    return render_template('bottom.html')

@app.route('/getdataset')
def getdataset():
    get_data = get_dataset() 
    return render_template('getdataset.html', get_data=get_data)


@app.route('/traindataset')
def traindataset():
    train_data = train_dataset() 
    return render_template('traindataset.html', train_data=train_data)


@app.route('/testdatasetknn')
def testdatasetknn():
    knn_data = test_dataset_knn()  
    return render_template('testdatasetknn.html', knn_data=knn_data)

@app.route('/testdatasetsvm')
def testdatasetsvm():
    svm_data = test_dataset_svm()  
    return render_template('testdatasetsvm.html', svm_data=svm_data)

@app.route('/processtestdata')
def processtestdata():
    pt_data = process_test_data()  
    return render_template('processtestdata.html', pt_data=pt_data)

@app.route('/displayfraudusers')
def displayfraudusers():
    fraudulent_users = display_fraud_users()  
    return render_template('displayfraudusers.html', fraudulent_users=fraudulent_users)


@app.route('/showaccuracy')
def showaccuracy():
    ac = show_accuracy()  
    return render_template('showaccuracy.html', ac=ac)

@app.route('/finduser', methods=['POST'])
def finduser():
    user_id = request.form['userid']
    user_info=find_user(user_id)
    return render_template('finduser.html', user_info=user_info)

if __name__ == '__main__':
    app.run(debug=True)
