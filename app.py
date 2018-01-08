from flask import Flask, request, render_template
from build_model import TextClassifier
import pickle

app = Flask(__name__)

with open('static/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    txt = request.form['txt']
    prediction = model.predict([txt])
    return prediction[0]


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
