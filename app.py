from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('random_forest_regression_Advertize_model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_increase_in_sale():
    TV = float(request.form.get('TV'))
    radio = float(request.form.get('radio'))
    newspaper = float(request.form.get('newspaper'))

    # prediction
    result = model.predict(np.array([TV,radio,newspaper]).reshape(1,3))
    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8082)

print(__name__)