from flask import Flask, render_template, request,jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,Activation,MaxPooling2D
from keras.utils.np_utils import normalize
from keras.layers import Concatenate
from keras import Input
import keras.backend as K

import time

img_size=100

app=Flask(__name__) 

def load_model():

	K.clear_session()

	model = Sequential()


	model.add(Conv2D(256,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(128,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))


	model.add(Conv2D(64,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(32,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(256,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2,input_dim=256,activation='sigmoid'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	model.load_weights('model/model-013.h5')
	return model

label_dict={0:'Covid19 Negative', 1:'Covid19 Positive'}

#convt to gray
def preprocess(img):

	img=np.array(img)

	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	#4D to pass into the NN
	reshaped=resized.reshape(1,img_size,img_size)
	return reshaped

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/predict", methods=["POST"])

def predict():
	print("hello")

	model=load_model()
	#time.sleep(60)
	print("hello")

	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)


	prediction = model.predict(test_image)
	result=np.argmax(prediction,axis=1)[0]
	accuracy=float(np.max(prediction,axis=1)[0])

	label=label_dict[result]

	print(prediction,result,accuracy)

	response = {'prediction': {'result': label,'accuracy': accuracy}}

	return jsonify(response)

app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">