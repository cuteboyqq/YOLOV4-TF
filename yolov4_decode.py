#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:59:07 2022

@author: ali
"""
import cv2
import random
import colorsys
import numpy as np

'''
func: filter_boxes_NonTF
1.filter boxes that score < score_threshold
2. convert xywh into y1x1y2x2 
'''
def filter_boxes_NonTF(box_xywh, scores, score_threshold=0.4, input_shape = 416.0):
    SHOW_LOG = True
    use_tf_nms = True
    scores_max =  []
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            max_val =  max(scores[i][j])
            scores_max.append(max_val)
    
    mask = []
    for i in range(len(scores_max)):
        if scores_max[i] >=  score_threshold:
            mask.append(True)
        else:
            mask.append(False)
     
    class_boxes = []
    pred_conf = []
    for i in range(len(box_xywh)):
        for j in range(len(box_xywh[i])):
            if mask[j]:
                class_boxes.append(box_xywh[i][j]) #append will have empty error
                pred_conf.append(scores[i][j]) #append will have empty error
                '''
                class_boxes_line = []
                pred_conf_line = []
                for k in range(len(box_xywh[i][j])):
                    class_boxes_line.append(box_xywh[i][j][k])
                class_boxes.append(class_boxes_line)
                for k in range(len(scores[i][j])):
                    pred_conf_line.append(scores[i][j][k])
                pred_conf.append(pred_conf_line)
                '''
    box_xy = []
    box_wh = []
    for i in range(len(class_boxes)):
        box_xy.append(class_boxes[i][:2])
        box_wh.append(class_boxes[i][2:])
                 
    box_yx = []
    box_hw = []
    for i in range(len(box_xy)):
       box_yx.append(box_xy[i][::-1])
       box_hw.append(box_wh[i][::-1])
    
    box_mins  = [ (box_yx[i] - (box_hw[i]/2.0) ) / np.array(input_shape) for i in range(len(box_hw))]
    box_maxes = [ (box_yx[i] + (box_hw[i]/2.0) ) / np.array(input_shape) for i in range(len(box_hw))]
    
    boxes = []
    for i in range(len(box_mins)):
        boxes.append([box_mins[i][0],box_mins[i][1],box_maxes[i][0],box_maxes[i][1]])
    
    print('len(boxes) = {}'.format(len(boxes)))
    if len(boxes)==0:
        boxes = [[[0.0,0.0,0.0,0.0]]]
        pred_conf = [[[0.0,0.0,0.0]]]
    else:
        if use_tf_nms:
            boxes = [boxes]
            pred_conf = [pred_conf]
        else:
            boxes = boxes
            pred_conf = pred_conf
    if SHOW_LOG:
        print('scores: \n {}\n shape {}'.format(scores,scores.shape))
        print('scores_max.shape: \n',np.shape(scores_max))
        #------------------------------------------------------------
        print('mask.shape: {}'.format(np.shape(mask)))
        #------------------------------------------------------------
        print('box_xywh:\n',box_xywh)
        print('box_xywh.shape:',box_xywh.shape)
        #-------------------------------------------------------------
        print('class_boxes:\n {}'.format(class_boxes))
        print('class_boxes.shape: {}'.format(np.shape(class_boxes)))
        print('pred_conf:\n {}'.format(pred_conf))
        print('pred_conf.shape: {}'.format(np.shape(pred_conf)))
        #---------------------------------------------------------
        print('box_xy:\n {}'.format(box_xy))
        print('box_xy shape:{}'.format(np.shape(box_xy)))
        print('box_wh:\n {}'.format(box_wh))
        print('box_wh shape:{}'.format(np.shape(box_wh)))
        print('input_shape: {}'.format(input_shape))
        #---------------------------------------------------------
        print('box_yx:\n {}'.format(box_yx))
        print('box_hw:\n {}'.format(box_hw))
        #---------------------------------------------------------
        print('box_mins :\n {}'.format(box_mins))
        print('box_maxes :\n {}'.format(box_maxes))
        print('box_mins & box_maxes : \n')
        for i in range(len(box_mins)):
            print(box_mins[i])
            print(box_maxes[i])
        #---------------------------------------------------
        print('boxes : \n {}'.format(boxes))
        print('boxes.shape \n {}'.format(np.shape(boxes)))
        print('pred_conf :\n {}'.format(pred_conf))
        print('pred_conf.shape: \n {}'.format(np.shape(pred_conf)))
   
    
    return (boxes, pred_conf)

'''
Get union of  picked and  box
input:
    box : [x1,y1,x2,y2]
    picked : [x1,y1,x2,y2]
output:
    union: (list) union of  picked and  box [x1,y1,x2,y2]
Purpose :update nms picked box by union of picked and removed box
'''             
def Get_union_box(box,picked):
    union =  [min(box[0],picked[0]), min(box[1],picked[1]), max(box[2],picked[2]), max(box[3],picked[3])]
    return union
''' 
Get the iou of picked_box and box
input:
    box : [x1,y1,x2,y2]
    picked : [x1,y1,x2,y2]
output:
    iou: (float)iou of  picked and box  
'''
def iou(box,picked,img_size=416): 
    
    #print('iou\n')
    #print('box: \n {}'.format(box))
    #print('picked: \n {}'.format(picked))
    box_area = (box[2]-box[0])*(box[3]-box[1])*img_size*img_size
    picked_area = (picked[2]-picked[0])*(picked[3]-picked[1])*img_size*img_size
    inter = [max(box[0],picked[0]), max(box[1],picked[1]), min(box[2],picked[2]), min(box[3],picked[3])]
    
    inter_area = (inter[2]-inter[0])*(inter[3]-inter[1])*img_size*img_size
    inf = 0.00001
    iou = inter_area / (box_area+picked_area-inter_area + inf)
    return iou

'''
Implement NMS by pure python mainly using numpy
    input:
        boxes : list of boxes [[x1,y1x2,y2],[x1,y1x2,y2],...]
        pred_conf : list of class confidence [[0.57,0.15,0.05,...],[0.24,0.85,0.14,...]...]
        num_classes : number of classes (int)
        iou_threshold : iou th
        use_tf_nms : True/False : Use/Not Use Tensorflow nms function
        img_size : (int) inference image size
    output:
        picked : BB after doing nms
        picked_pred_conf : class confidence after doing nms
        picked_label : equal to np.argmax(picked_pred_conf), picked label after doing nms
'''
def Combined_Non_Max_Suppression_NonTF(boxes,pred_conf,num_classes=3,iou_threshold=0.5,use_tf_nms=True,img_size=416):
    if not use_tf_nms:
        labels = []
        for i in range(len(pred_conf)):
            label = np.argmax(pred_conf[i])
            labels.append(label)
        
        picked = []
        picked_label = []
        picked_pred_conf = []
        for clas in range(num_classes):
            for j in range(len(boxes)):
                keep = True
                if not labels[j]==clas:
                    continue
                for k in range(len(picked)):
                    if not picked_label[k]==clas:
                        continue
                    if iou(boxes[j],picked[k],img_size)>iou_threshold:
                        picked[k] = Get_union_box(boxes[i][j],picked[k])#update picked_box by union of box and picked_box
                        keep = False# if iou of any of picked box and box > th, then remove this box   
                        break
                    else:
                        keep = True
                if keep==True:
                    picked.append(boxes[j])
                    picked_label.append(labels[j])
                    picked_pred_conf.append(pred_conf[j])
    else:
        labels = []
        for i in range(len(pred_conf)):
            for j in range(len(pred_conf[i])):
                label = np.argmax(pred_conf[i][j])
                labels.append(label)
        #print('labels: \n {}\n shape:{}'.format(labels,np.shape(labels)))
        
        picked = []
        picked_label = []
        picked_pred_conf = []
        for clas in range(num_classes):
            for i in range(len(boxes)):
                for j in range(len(boxes[i])): 
                    keep = True
                    if not labels[j]==clas:
                        #print('not class')
                        continue
                    for k in range(len(picked)):
                        if not picked_label[k]==clas:
                            #print('picked_label[k] is not class {}'.format(k))
                            continue
                        
                        if iou(boxes[i][j],picked[k],img_size)>iou_threshold:
                            picked[k] = Get_union_box(boxes[i][j],picked[k]) #update picked_box by union of box and picked_box
                            keep = False
                            break # if iou of any of picked box and box > th, then remove this box   
                        else:
                            #print('boxes[i][j]: \n {}'.format(boxes[i][j]))
                            #print('iou: {}\n'.format(iou(boxes[i][j],picked[k],img_size)))
                            keep = True
                    if keep==True:
                        print('Add picked {}'.format(boxes[i][j]))
                        picked.append(boxes[i][j])
                        picked_label.append(labels[j])
                        picked_pred_conf.append(pred_conf[i][j])
                
    return picked, picked_pred_conf, picked_label


'''Get class names by class file'''
def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


''' 
Draw bbox(bbox is obtained by doing NMS function) on the image
    input : image --> The image you want to draw BB
            bboxes --> The bboxes list 
            classes : class list
            show_label : bool True/False : Enable/Disable show label
    output :
            image :   image with BB 
'''
def draw_bbox_NonTF(image, bboxes, classes=read_class_names("/home/ali/YOLOV4-TF/data/classes/factory.names"), show_label=True):
    
    use_tf_version = True
    
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    
    print('image_h: {}, image_w: {}'.format(image_h,image_w))
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)
    
    picked, picked_pred_conf, picked_label = bboxes
    print('in draw_bbox_NonTF:')
    print('picked = {}'.format(picked))
    print('picked_pred_conf = {}'.format(picked_pred_conf))
    print('picked_label = {}'.format(picked_label))
    
    if use_tf_version:
        for i in range(len(picked)):
            #for j in range(len(picked[i])):
            coor = picked[i]
            new_coor = [0.0,0.0,0.0,0.0]
            print('coor : \n {}'.format(coor))
            print('before coor[0]:{}, coor[1]:{}, coor[2]:{}, coor[3]:{}'.format(coor[0],coor[1],coor[2],coor[3]))
            '''
            if int(coor[0] * image_h) > 0:
                new_coor[0] = int(coor[0] * image_h)
            else:
                new_coor[0] = 0
                
            if int(coor[2] * image_h) > 0:
                new_coor[2] = int(coor[2] * image_h)
            else:
                new_coor[2] = 0
                
            if int(coor[1] * image_w) > 0:
                new_coor[1] = int(coor[1] * image_w)
            else:
                new_coor[1] = 0
                
            if int(coor[3] * image_w) > 0:
                new_coor[3] = int(coor[3] * image_w)
            else:
                new_coor[3] = 0
            '''
            
            new_coor[0] = int(coor[0] * image_h) 
            new_coor[2] = int(coor[2] * image_h) 
            new_coor[1] = int(coor[1] * image_w) 
            new_coor[3] = int(coor[3] * image_w) 
            
            print('after coor[0]:{}, coor[1]:{}, coor[2]:{}, coor[3]:{}'.format(new_coor[0],new_coor[1],new_coor[2],new_coor[3]))
            fontScale = 0.5
            print('i={}'.format(i))
            '''
            for i in range(len(picked_label)):
                print(picked_label[i])
            '''
            print('picked_label[i]: {} \n'.format(picked_label[i]))
            score = max(picked_pred_conf[i]) #fake data
            class_ind = np.argmax(picked_pred_conf[i])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (int(new_coor[1]), int(new_coor[0])) , (int(new_coor[3]), int(new_coor[2])) #1,0,3,2
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            
            if show_label:
                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = ( int(c1[0] + t_size[0]), int(c1[1] - t_size[1] - 3))
                cv2.rectangle(image, c1, (np.int32(c3[0]), np.int32(c3[1])), bbox_color, -1) #filled

                cv2.putText(image, bbox_mess, (int(c1[0]), np.int32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    
    
    return image



