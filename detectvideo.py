import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4_decode import *
from core.yolov4 import filter_boxes,filter_boxes_NonTF
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints_yolov4_20220808_ciou_tf25_mosaic_aug_best/yolov4-best-416','path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', '/home/ali/factory_video/ori_video_ver2.mp4', 'path to input video')
flags.DEFINE_float('iou', 0.10, 'iou threshold')
flags.DEFINE_float('score', 0.40, 'score threshold')
flags.DEFINE_string('output', './inference/ori_video_infer_20220809_aug_mosaic_9min_score_0_40_finetune.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dis_cv2_window', False, 'disable cv2 window during the process') # this is good for the .ipynb

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print('input_details:\n',input_details)
        print('output_details:\n',output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            print("Get frame success !")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")
        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            #print('pred:\n {}'.format(pred))
            #print('prd[0]:\n',pred[0],'\n',pred[0].shape) #1x26x26x24
            #print('prd[1]:\n',pred[1],'\n',pred[1].shape) #1x13x13x24
            #print('prd[2]:\n',pred[2],'\n',pred[2].shape) #1x52x52x24
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                #boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                #input_shape=tf.constant([input_size, input_size]))
                boxes, pred_conf = filter_boxes_NonTF(pred[1], pred[0], score_threshold=0.25,input_shape=416.0)
            else:
                #boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                #input_shape=tf.constant([input_size, input_size]))
                boxes, pred_conf = filter_boxes_NonTF(pred[0], pred[1], score_threshold=0.25,input_shape=416.0)
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            #print(len(pred_bbox.items()))
            for key, value in pred_bbox.items():
                #print(key)
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
                #print("boxes:\n",boxes)
                #print("pred_conf:\n",pred_conf)
        #print('boxes:\n{}'.format(boxes))
        #print('pred_conf:\n{}'.format(pred_conf))
        print('before NMS')
        boxes = boxes.numpy()
        pred_conf = pred_conf.numpy()
        print('boxes:\n{} \n boxes.shape {}'.format(boxes,boxes.shape))
        print('pred_conf:\n{} \n pred_conf.shape {}'.format(pred_conf,pred_conf.shape))
        '''Alister add NMS 2022-08-08'''        
        picked, picked_pred_conf, picked_label = Combined_Non_Max_Suppression_NonTF(boxes,pred_conf,num_classes=3,iou_threshold=0.10,use_tf_nms=True)
        print('picked:\n {} \n picked_pred_conf:\n {} \n picked_label:\n {} \n'.format(picked, picked_pred_conf, picked_label))
        '''==========================================================================================================================='''
        '''
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )'''
        #pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        pred_box = [picked, picked_pred_conf, picked_label]
        #image = utils.draw_bbox(frame, pred_bbox)
        image = draw_bbox_NonTF(frame, pred_box,show_label=True)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        print(info)

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not FLAGS.dis_cv2_window:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        if FLAGS.output:
            out.write(result)

        frame_id += 1

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
