from flask_restful import Resource
from flask import request
from app.models.vision import init_model, predict_model
from PIL import Image
import io

model, ds_info=init_model()

class PredictResource(Resource):
    def get(self):
        print("Solicitud Get a la Api")
    def post(self):
        print("Llamada Api")
        file=request.files["file"]
        image=Image.open(io.BytesIO(file.read()))

        prediction = predict_model(image, model, ds_info)
        
        return {"prediction":prediction}