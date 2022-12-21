import numpy as np 
import info 
import os
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import load_model
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
predictt=5

app = Flask(__name__)

MODEL_PATH = 'wasteclass_model.h5'
classes={
    0:'cardboard',
    1:'glass',
    2:'metal',
    3:'paper',
    4:'plastic'
}
model = load_model(MODEL_PATH)       
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    img = cv2.imread(img_path)
    display = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    display = tf.cast(display, tf.float32)
    predicted_classname = np.argmax(model.predict(np.array([display])))
    return predicted_classname


@app.route('/', methods=['GET'])
def index():
    return render_template('landing.html')
@app.route('/input', methods=['GET'])
def input():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predicted_classname = model_predict(file_path, model)
        print(predicted_classname)
        prediction=f"Predicted Image as {classes[predicted_classname]}"
        global predictt
        predictt=classes[predicted_classname]
        return prediction
    else:
        return render_template("landing.html")        
@app.route('/output', methods=['GET'])
def output():
    global predictt
    x=eval(predictt)
    name1=x["name1"]
    pno1=x["pno1"]
    add1=x["add1"]
    name2=x["name2"]
    pno2=x["pno2"]
    add2=x["add2"]
    name3=x["name3"]
    pno3=x["pno3"]
    add3=x["add3"]

    return render_template('output.html',prediction=predictt,name1=name1,pno1=pno1,add1=add1,name2=name2,pno2=pno2,add2=add2,name3=name3,pno3=pno3,add3=add3)

if __name__ == '__main__':
    app.run(debug=True)