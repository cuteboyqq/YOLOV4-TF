#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:49:04 2022

@author: ali
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import time






def create_category_index(label_path='classes.txt'):
    f = open(label_path)
    category_index = {}
    for i, val in enumerate(f):
        if i != 0:
            val = val[:-1]
            if val != '???':
                category_index.update({(i-1): {'id': (i-1), 'name': val}})
            
    f.close()
    return category_index


#Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path="./runs/train/exp/weights/best-int8.tflite")
#interpreter.allocate_tensors()

# Get input and output tensors.
#input_details = interpreter.get_input_details()
#output_details = interpreter.get_output_details()
#print(output_details)


category_index = create_category_index()



def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def YOLOdetect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    print('output_data.shape : {}'.format(np.shape(output_data)))
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    #boxes = (output_data[..., :4])    # boxes  [25200, 4]
    print('boxes shape : {}'.format(np.shape(boxes)))
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    print('scores shape : {}'.format(np.shape(scores)))
    classes = classFilter(output_data[..., 5:]) # get classes
    print('classes shape : {}'.format(np.shape(classes)))
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    #y, x, h, w = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    #print('{},{},{},{}'.format(x,y,w,h))
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]
    #xyxy = [y , x , h, w]  # xywh to xyxy   [4, 25200]
    for i in range(4):
        print(xyxy[i])
    
    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]


framework = 'tflite'
video_path = "/home/ali/factory_video/ori_video_ver2.mp4"
vid = cv2.VideoCapture(video_path)
output = "/home/ali/YOLOV5/factory_infer_2min.mp4"
output_format = "XVID"
input_size = 448
dis_cv2_window = False
if framework == 'tflite':
    interpreter = tf.lite.Interpreter(model_path="./runs/train/exp/weights/best-int8.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    print('input_details:\n',input_details)
    print('output_details:\n',output_details)
#else:
    #saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    #infer = saved_model_loaded.signatures['serving_default']
if output:
    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*output_format)
    out = cv2.VideoWriter(output, codec, fps, (width, height))

frame_id = 0
while True:
    return_value, frame = vid.read()
    if return_value:
        print("Get frame success !")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(frame)
        #image = frame.astype(np.uint8)
    else:
        if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
            print("Video processing complete")
            break
        raise ValueError("No image! Try with another video format")
    
    frame_size = frame.shape[:2]
    #image_data = cv2.resize(image, (input_size, input_size))
    #image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    #image_data = image_data / 255.
    #image_data = image_data[np.newaxis, ...].astype(np.uint8)
    
    
    #image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(frame, (input_size, input_size), cv2.INTER_AREA)
    image_data = image_data.reshape([1, input_size, input_size, 3])
    
    
    prev_time = time.time()
    
    """Output data"""
    interpreter.set_tensor(input_details[0]['index'], image_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # get tensor  x(1, 25200, 7)
    xyxy, classes, scores = YOLOdetect(output_data) #boxes(x,y,x,y), classes(int), scores(float) [25200]
    
    print(len(scores))
    for i in range(len(scores)):
        if ((scores[i] >= 0.4) and (scores[i] <= 1.0)):
            H = frame.shape[0]
            W = frame.shape[1]
            
            xmin = int(max(1,(xyxy[0][i]*float(W/448.0) )))
            ymin = int(max(1,(xyxy[1][i]*float(H/448.0) )))
            xmax = int(min(W,(xyxy[2][i]*float(W/448.0) )))
            ymax = int(min(H,(xyxy[3][i]*float(H/448.0) )))
            
        
            print('xmin:{}, ymin:{},xmax:{},ymax:{}'.format(xmin,ymin,xmax,ymax))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
          
    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if not dis_cv2_window:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if output:
        out.write(result)