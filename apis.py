import boto3
import io
import h5py
from tensorflow import keras
import numpy as np
import pickle

classes={'Bacterial leaf blight': 0, 'Brown spot': 1, 'Leaf smut': 2}

s3_client=boto3.client('s3')
def upload_np_to_s3(image_np,transaction_id=None):
    if transaction_id is None:
        transaction_id='temp1011'
    print("tid",transaction_id)
    array_data=io.BytesIO()
    pickle.dump(image_np, array_data)
    array_data.seek(0)
    s3_client.upload_fileobj(array_data,'btpimages1081',transaction_id)
    # res=io.BytesIO()
    # s3_client.download_fileobj('btpimages1081',transaction_id,res)
    # res.seek(0)
    # res=pickle.load(res)
    # print(res)

def predict_image_class(image_np):
    model=keras.models.load_model("file_name.h5")
    model.summary()
    prediction=model.predict(image_np)
    for a in classes:
        if(classes[a]==prediction):
            return a
    return ""

image_np=np.random.randn(10)
upload_np_to_s3(image_np=image_np)