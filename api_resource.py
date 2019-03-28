# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:46:28 2019

@author: Mathu_Gopalan
"""


from flask_restful import Resource,reqparse
from infer import InferImage

#import logging

#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger('HELLO WORLD')

class Image(Resource):
    '''
    Resources for the flask api are defined in class
    Method: post, accepts the inputfile path
    '''
    
    parser = reqparse.RequestParser()
    parser.add_argument('path',
        type=str,
        required=True,
        help="This field cannot be left blank!"
    )
    
    def post(self):
        '''
        post method, accepts path as header object
        '''
        data = Image.parser.parse_args()
        folder_path = data['path']
        #write code to call inferpy class
        result = Image.tf_infer(folder_path)
        print (f"In Imageclass, result as {result}")
        return result,200
   
    @classmethod
    def tf_infer(cls, path):
        predicts=InferImage.infer(path)
        return predicts
        
        
        
        
    