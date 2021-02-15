from flask import render_template, request, redirect, url_for, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
from PIL import Image, ImageFile
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSION = set(['png','jpg','jpeg','gif'])

def base():
    return render_template('base.html')

def index():
    return render_template('index.html')

def faceapp():
    return render_template('faceapp.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSION

def getwidth(path):
    img = Image.open(path)
    size = img.size # width and height
    aspect = size[0]/size[1] # width / height
    w = 300 * aspect
    return int(w)

def prepare_image(path):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def gender():

    if 'image' not in request.files:
        return render_template('gender.html', fileupload=False, prediction='')

    file = request.files['image']
    if file.filename =='':
        return render_template('gender.html', fileupload=False, prediction='No posted image. Please upload an image.')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print("***"+filename)
        path = os.path.join(UPLOAD_FOLDER, filename) # get the file path
        file.save(path)  # save the file to the specified upload folder
        w = getwidth(path)
        preprocessed_image = prepare_image(path)
        mobile = tf.keras.applications.mobilenet.MobileNet()
        predictions = mobile.predict(preprocessed_image)
        results = imagenet_utils.decode_predictions(predictions)
        items = []
        for item in results[0][:3]:
            items.append({item[1]: round(float(item[2]),4)})

        return render_template('gender.html', fileupload=True, img_name=filename, w=w, prediction='The image is most likely {}'.format(items))
    else:
        return render_template('gender.html', fileupload=False, prediction='Invalid File extension. Please upload an image with extension png, jpg, jpeg or gif')
