from flask import Flask, render_template, request
import numpy as np
import pickle

# Load the model
model = pickle.load(open("Breast_Cancer_Classification/model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['feature']
    features_list = [float(x) for x in features.split(',')]
    np_features = np.array(features_list, dtype=np.float32).reshape(1, -1)
    
    pred = model.predict(np_features)
    output = ["Cancer" if pred[0] == 1 else "No Cancer"]
    
    return render_template('index.html', message=output)

if __name__ == '__main__':
    app.run(debug=True)
