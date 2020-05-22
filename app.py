import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

scoreurl = 'http://a33f08e4-a26a-4f35-8265-6d72c3a2ecb3.westeurope.azurecontainer.io/score'

def scaling(test):
    file = pd.read_csv('data/turbine_training_set.csv')
    file = file[['Turbine_exit_pressure_bar', 'Turbine_exit_temperature_C', 'Ship_speed_knots' ]]
    sc = MinMaxScaler()
    train = sc.fit_transform(file)
    test = sc.transform(test)
    return test
      

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/prediction',methods = ['POST'])
def prediction():
    scoreurl = 'http://a33f08e4-a26a-4f35-8265-6d72c3a2ecb3.westeurope.azurecontainer.io/score'
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    scaled = scaling(final_features)
    
    test = json.dumps({'data': scaled.tolist()})
    print(test)
    print(scaled)

    print('reached the scaling part')
    headers = {'Content-Type':'application/json'}


    resp = requests.post(scoreurl, test, headers=headers)
    resp = json.loads(json.loads(resp.text))
    print(type(resp))
    print('Normal text is: ',resp)
    print('Inverse transform is',resp)
    print(resp['result'])
    result = inverse_transform(resp['result'])
    
 

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Turbine Output is: {}".format(result[0][0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    scaled = scaling(data)
    headers = {'Content-Type':'application/json'}
    resp = requests.post(scoreurl, scaled, headers=headers)
    
    print(resp)
    print(resp.text)
    
    
    
    #prediction = model.predict([np.array(list(data.values()))])

    output = resp[0]
    return jsonify(output)



if __name__ == '__main__':
    print('starting the app')
    app.run(debug=True, use_reloader=False )