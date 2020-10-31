from __future__ import division, print_function
# coding=utf-8
import sys
import os
import shutil
import glob
import re
import numpy as np
import tensorflow as tf
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
from flask_cors import CORS
app = Flask(__name__)
CORS(app) # This will enable CORS for all routes

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

def delete_file():
    if os.path.exists("demofile.txt"):
        os.remove("demofile.txt")
    else:
        print("The file does not exist")


def model_predict(img_path, model):
    # print(img_path)
    # # img = image.load_img(img_path, target_size=(100, 100))
    # print(type(img))
    # print(img)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.)
    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)
    test_generator = test_datagen.flow_from_directory("./uploads",
                                                    batch_size = 4,
                                                    class_mode = None, 
                                                    target_size = (100, 100))
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')
    test_generator.reset()
    preds = model.predict(test_generator)
    print("preds",preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

class_names = ['lantana', ' parthenium', 'prosopis']

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    global class_names
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/zz', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        shutil.rmtree('./uploads/zz')
        os.mkdir('./uploads/zz')
        
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        predicted_class_indices  = np.argmax(preds,axis=1)
        # print("predicted indices yididiyaaa mikiii des yebelachuh: betam yikirta///des beluachewal", predicted_class_indices )
        res = class_names[predicted_class_indices[0]]
        print('resss====',res)
        if (res == 'lantana'):
            result = 'lantana weed (Lantana camara). \n Lantana ((lan-tan-uh)) is a genus of about 150 species of perennial flowering plants in the verbena family, Verbenaceae. They are native to tropical regions of the Americas and Africa \n Herbicides that can be used are : leaf mining Beatles | Picloram + 2,4-D amine '
            print('res: lantana')
        elif (res == 'parthenium'):
            result  =  ' parthenium weed (Parthenium hysterophorus). \n Parthenium hysterophorus is a species of flowering plant in the aster family, Asteraceae. It is native to the American tropics. \n Herbicides that can be used are : | 2,4-D | Picloram + 2,4-D amine  | Glyphosate + metsulfuron | Paraquat + diquat'
            print('res: lantana')
        elif(res == 'prosopis'):
            result = 'prosopis (Prosopis juliflora). \n Prosopis juliflora (Spanish: bayahonda blanca, Cuji [Venezuela], Aippia [Wayuunaiki]) is a shrub or small tree in the family Fabaceae, a kind of mesquite. \n Herbicides that can be used are : Populare | phosphoro curale'
           
        return res
    return None


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.run(host='0.0.0.0',debug=True, port=port)

