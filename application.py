from flask import Flask,request,Response
from tensorflow import keras
from keras.preprocessing import image
from io import BytesIO
from PIL import Image
import jsonpickle
import numpy as np
import cv2
from apis import predict_image_class
application=Flask(__name__)

@application.route('/process_image', methods=['POST', 'GET'])
def process_image():
    print("started")
    image_path=request.files['image']
    
    if(image_path.filename!=''):
        #img=misc.imread(image_path)
        image_nparr=image.img_to_array(Image.open(image_path))
        image_nparr = np.expand_dims(image_nparr, axis = 0)
        #image_path.save(image_path.filename)
    print(image_nparr)
    prediction=predict_image_class(image_nparr)
    print("prediction is ",prediction)
    print(type(image_nparr))
    res={
        'plant_name':'b',
        'disease_detected':prediction,
        'more_info':"NA"
    }
    return Response(res, status=200, mimetype='application/json')

if __name__=='__main__':
    application.run(debug=True)