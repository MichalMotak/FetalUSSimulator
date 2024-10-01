from peaceful_pie.unity_comms import UnityComms
import argparse

import time

import numpy as np


import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io

import simplejpeg

from PIL import Image
import base64
import requests
import os

ID_SAVED = []
FILE_SAVE_PATH = "/home/mmotaksano/Unity_projects/FetalUltrasoundSimulator_test1/AcquiredData/Images1_rot_all/"


STOP_ITERATOR = 0

def run(uc):
    print("args ")
    

    unity_output = uc.GetOutput()
    shape = unity_output['Item1']
    res = unity_output['Item2']
    id = unity_output['Item3']
    poses = unity_output['Item4']

    if id in ID_SAVED:
        return 0
    

    ID_SAVED.append(id)



    print("ID ", id)
    print("res ", type(res))

    print("res ", res[:100])
    print("res ", len(res))
    print("shape ", shape)
    print(res.encode()[:50])
    print(len(res.encode()))
    # out_img = uc.GetImage()
    print("poses ", poses)


    # 
    byte_stream = base64.b64decode(res.encode())
    print(byte_stream[:50])

    # image = np.vstack(res).astype(np.float32)
    # # image 
    # print("image ", image.shape)
    # print("image ", image)
    # print(image.dtype)
 

    # plt.imshow(image)
    # plt.show()


    # Decode JPEG byte array

    # img = simplejpeg.decode_jpeg(res)
    # img = cv2.imdecode(np.fromstring(res, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # img = cv2.imdecode(np.frombuffer(res.encode()), cv2.IMREAD_UNCHANGED)                
    # 
    # 
    #                            
    # 
    # 
    # 
    # 
    #    
    # image = Image.open(io.BytesIO(res.encode()))
    # d = np.frombuffer(res.encode(),dtype=np.uint8)
    # print(type(d), len(d), d)
    # # # image.show()

    # # print("image ", img)
    # print(type(res.encode()))

    # print(simplejpeg.is_jpeg(res.encode()))
    # img = simplejpeg.decode_jpeg(data = res.encode(), colorspace = "RGB")
    # print(img.shape)



    # PIL 

    # image = Image.frombytes('RGB', (211,232), res.encode(), 'raw')
    # image.show()




    image = Image.open(io.BytesIO(byte_stream))
    image_file_name = f"image_{id}.png"
    print(FILE_SAVE_PATH + "/" + image_file_name)
    image.save(FILE_SAVE_PATH + image_file_name)
    # image.show()

    global STOP_ITERATOR
    STOP_ITERATOR += 1



if __name__ == "__main__":

    
    print("d")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    uc = UnityComms(port = args.port)


    for i in range(0,2000000):

        run(uc)

        print("STOP_ITERATOR ", STOP_ITERATOR)

        time.sleep(0.25)



    # for i in range(0,25):

    #     try:
    #         run(uc)

    #         time.sleep(0.1)
    #     except requests.exceptions.ConnectionError:
    #         print("aa")










# from peaceful_pie.unity_comms import UnityComms
# import argparse

# def run(args: argparse.Namespace) -> None:
#     print(args)
#     uc = UnityComms(port = args.port)

#     uc.Say(message = args.message)
     

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--message", type = str, required=True)
#     parser.add_argument("--port", type=int, default=9000)
#     args = parser.parse_args()
#     run(args)