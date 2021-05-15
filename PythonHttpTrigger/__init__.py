import logging
import sys
import os
#from pydrive.drive import GoogleDrive
import azure.functions as func
import fastai
# from fastai.vision import *
from fastai.vision.all import *
#from fastai.vision.widgets import *
#from fastai.imports import *
#import requests
from io import BytesIO
import base64, re
from PIL import Image
from PythonHttpTrigger.prediction import *
from PythonHttpTrigger.imageconverter import *



############ Custom functions##########

##########################

#Windows path
#path = 'C:\\Users\\Christian\\source\\repos\\pythonAzureFunction\\PythonHttpTrigger'
#Universal path
connect_str = "DefaultEndpointsProtocol=https;AccountName=pythonfunctiontesting;AccountKey=6MDBqMjlSTIzvdG4hr6+3i/XI3ZVAmsrl/xVPiOd7UTslHYZSKn4OlknpWV9nAeGmmoBqClx8Rt8OFwGtNHS5A==;EndpointSuffix=core.windows.net"

containerName = '4298b4ec-b6c6-47dc-97f2-658d6b2dc2d1'

fileName = 'No_transform_LevelFinderV3.pth'

path = os.path.join(Path.cwd(),'PythonHttpTrigger')

## Runs on post ####
def main(req: func.HttpRequest) -> func.HttpResponse:
    

    logging.info('Python HTTP trigger function processed a request.')
    theLevel = 0
    recievedTensor = ''
    theimage = ''
    try:
        treq_body  =  req.get_json()
        image2 = treq_body.get('image')
        name = treq_body.get('name')
    except:
        image2 = 'ERROR'
        name = 'ERROR'
        print("###########################")
        print(f'¤¤¤¤¤¤\t{name}\t¤¤¤¤¤¤')
        print("###########################")
        pass
    if image2 != 'ERROR' and image2 != None:
        recvImg = base64toimage(image2)
        #theimage = recvImg
        recievedTensor  = RunPrediction(recvImg)  
        base64fromTensor = tensor2image(recievedTensor)  
        theLevel = checkLevel(recievedTensor)
        theimage = base64fromTensor
        print("###########################")
        print(f't¤¤¤¤¤¤¤\t{theLevel}\t¤¤¤¤¤¤¤')
        print("###########################")
    
    if name:
        if name.lower() == 'bertil':# or name == 'Bertil':
            response = (f"Bertil mode.. This is a tribute to {name}. Your effort is valued at {theLevel} % ")
            jsonObj = json.dumps({"name":str(response),"tensor":str(recievedTensor),"image":theimage})
            return func.HttpResponse(
                jsonObj,
                status_code=200
            )
        else:    
            #return func.HttpResponse(f"Hello, {name}. This HTTP triggered PYTHON function executed successfully.")
            #tensorOut = f'Recieved tensor {re}'
            response = f'Hello, {name}. The calculated level is : {theLevel} % of the container'
            jsonObj = json.dumps({"name":str(response),"tensor":str(recievedTensor),"image":theimage})
            
            return func.HttpResponse(
                jsonObj,
                status_code=200
            ) #f" Hello, {name}. The calculated level is : {theLevel} %")
    else:
        return func.HttpResponse(
             "This HTTP triggered PYTHON function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )

