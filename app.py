import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)
#model = pickle.load(open('randomForestRegressor.pkl','rb'))

def scaling(data):
    
    sc = MinMaxScaler()
    values = sc.transform(final_features)
    return values



@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, scaling(final_features), headers=headers)

    prediction = resp
    print(prediction[0])

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Output Power for the Turbine {}".format(prediction[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)