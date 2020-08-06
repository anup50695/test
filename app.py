from flask import Flask, render_template, request
import pandas as pd
import json, requests
#from PIL import Image
app = Flask(__name__)
from azure.storage.blob import ContainerClient
from flask import request, send_from_directory
from datetime import datetime
from sqlalchemy import create_engine, event
from urllib.parse import quote_plus
import cv2
import numpy as np
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

### Prediction API details
project_id = "52d415a7-9e69-425d-bcb5-e173a0467ba8"
ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"
# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": "c00c9b7ff50c462aaf70dbb11aac60a8"})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('app_bootstrap.html')

@app.route('/detect_img', methods = ['GET', 'POST'])
def detect_img():
   if request.method == 'POST':
       img = request.files['file'].read()

       results = predictor.detect_image(project_id, "roadDamageModel", img)

        # compile results
       pred_res = []
       for prediction in results.predictions:
           pred_res.append([prediction.tag_name, prediction.probability * 100,
                            prediction.bounding_box.left, prediction.bounding_box.top,
                            prediction.bounding_box.width, prediction.bounding_box.height])

       pred_res = pd.DataFrame(pred_res)
       pred_res.columns = ["tag_name", "prob", "left", "top", "width", "height"]
       pred_res = pred_res[pred_res.prob > 40]
       pred_res = pred_res.drop_duplicates(subset="tag_name")
       tag = pred_res.tag_name.iloc[0]

      # add bounding boxes
       # Summarize the results.
       pred_res = []
       for prediction in results.predictions:
           pred_res.append([prediction.tag_name, prediction.probability * 100,
                            prediction.bounding_box.left, prediction.bounding_box.top,
                            prediction.bounding_box.width, prediction.bounding_box.height])

       pred_res = pd.DataFrame(pred_res)
       pred_res.columns = ["tag_name", "prob", "left", "top", "width", "height"]
       pred_res = pred_res[pred_res.prob > 40]
       pred_res = pred_res.drop_duplicates(subset="tag_name")

       # read image
       npimg = np.fromstring(img, np.uint8)
       # convert numpy array to image
       img_cv2 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
       cv2.imwrite("img_cv2.jpg", img_cv2)
       img_cv2 = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)

       for i in range(pred_res.shape[0]):
           x = int(pred_res.left.iloc[i] * img_cv2.shape[0])
           y = int(pred_res.top.iloc[i] * img_cv2.shape[1])

           x2 = x + int(pred_res.width.iloc[i] * img_cv2.shape[0])
           y2 = y + int(pred_res.height.iloc[i] * img_cv2.shape[1])

           img_cv2 = cv2.rectangle(img_cv2, (x, y), (x2, y2), (0, 0, 255), 2)
           tag_name = pred_res.tag_name.iloc[i]

           org = (x + 40, y + 40)
           img_cv2 = cv2.putText(img_cv2, tag_name, org, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                 fontScale=1, color=(255, 0, 0), thickness=2)
           img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        # write image with boudning boxes
       cv2.imwrite("model_pred.jpg",img_cv2_rgb)

       # write to Azure storage
       CONNECT_STR = "DefaultEndpointsProtocol=https;AccountName=azurehackml2012795721;AccountKey=ISNFmGieyfmUixhgZqrVCXdk4S0STTULUqfcLTJMpmWeFrU6u1FlyMIljJRp6+tx5/0KAqfqM78KZcbMxbLQ7w==;EndpointSuffix=core.windows.net"
       CONTAINER_NAME = "ai4pscapsteam"

       input_file_path = "model_pred.jpg"
       output_blob_name = "model_pred.jpg"

       container_client = ContainerClient.from_connection_string(conn_str=CONNECT_STR,
                                                                 container_name=CONTAINER_NAME)

       # Upload file
       with open(input_file_path, "rb") as data:
           container_client.upload_blob(name=output_blob_name, data=data, overwrite=True)

   model_pred_file_name = "https://azurehackml2012795721.blob.core.windows.net/ai4pscapsteam/model_pred.jpg"

   return render_template("pred_img.html",
                          model_pred_file_name = model_pred_file_name
                          )





if __name__ == '__main__':

    app.run(host='0.0.0.0', port=80)
