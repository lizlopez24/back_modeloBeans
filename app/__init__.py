from flask import Flask, send_from_directory
from flask_restful import Api
from app.resources.predict import PredictResource

def create_app():
    app=Flask(__name__, static_folder="../static")
    api=Api(app)
    
    #Rutas
    api.add_resource(PredictResource, "/modelo")

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    return app