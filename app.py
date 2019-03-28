# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:45:01 2019

@author: Mathu_Gopalan
"""
from flask import Flask, request, redirect, jsonify
from flask_restful import Api
from api_resource import Image

app = Flask(__name__)
api = Api(app)

api.add_resource(Image, '/upload')

if __name__=="__main__":
    app.run()