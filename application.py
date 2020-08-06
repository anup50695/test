from flask import Flask, render_template, flash, request,url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import pandas as pd
from werkzeug.utils import secure_filename
from azure.storage.blob import ContainerClient
import cv2

import numpy as np
#from werkzeug.contrib.cache import SimpleCache
#cache = SimpleCache()
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

### Prediction API details
project_id = "52d415a7-9e69-425d-bcb5-e173a0467ba8"
ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"
# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": "c00c9b7ff50c462aaf70dbb11aac60a8"})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)


pd.options.display.float_format = '{:,.2f}%'.format

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

class ReusableForm(Form):
    @app.route('/')
    def student():        
        return render_template('summary.html')
    
    @app.route('/search',methods = ['POST', 'GET'])
    def home():
        if request.method == 'POST':          
            
            return render_template("login-page.html")
        
    @app.route('/route1',methods = ['POST', 'GET'])
    def route1():
        return render_template("predict.html")
        # print('Here')
        # if request.method == 'POST':
        #     return render_template("predict.html")

    @app.route('/result',methods = ['POST', 'GET'])
    def result():
        if request.method == 'POST':
            result = request.form
            if result['UserId'] == 'admin@caps.com' and result['Password'] == 'admin': 
                
                
                return render_template("summary.html")
            else:
                error = "UserId or Password is incorrect. Please try again."
                return render_template("login-page.html",error = error)

    @app.route('/test', methods=['POST', 'GET'])
    def test():
        if request.method == 'POST':
            img = request.files['file'].read()
        return render_template("pred_img.html")

    @app.route('/searchTable',methods = ['POST', 'GET'])
    def sTable():
        if request.method == 'POST':
            result = request.form
            f = request.files['upImg']
            f.save(secure_filename(f.filename))
            print('Here')

    @app.route('/detect_img', methods=['GET', 'POST'])
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
            cv2.imwrite("static/img_cv2.jpg", img_cv2)
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
            cv2.imwrite("model_pred.jpg", img_cv2_rgb)

            #write to Azure storage
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
                               model_pred_file_name=model_pred_file_name,
                               tag = tag
                               )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
