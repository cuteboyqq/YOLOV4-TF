from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
#from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
from tensorflow.python.saved_model import tag_constants
import cv2
from core.yolov4 import filter_boxes
import glob
import json
from tqdm import tqdm
from core.yolov4 import YOLO, decode, filter_boxes
flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
#flags.DEFINE_string('weights', './models/checkpoints_yolov4_20220812_best/yolov4-best', 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_string('output', './models/checkpoints_yolov4_20220815/yolov4', 'path to output')
flags.DEFINE_string('output_best', './models/checkpoints_yolov4_20220815_best/yolov4-best', 'path to output')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.50, 'define score threshold')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_float('iou', 0.10, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_string('annotation_path', r'/home/ali/YOLOV4-TF/data/dataset/factory_data_val_20220806.txt', 'path to output')

def infer(batch_data, model):
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    # batch_data = tf.constant(image_data)
    feature_maps = model(batch_data)
    bbox_tensors = []
    prob_tensors = []
    if FLAGS.tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        FLAGS.framework)
            else:
                output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        FLAGS.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        FLAGS.framework)
            elif i == 1:
                output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        FLAGS.framework)
            else:
                output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,
                                        FLAGS.framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])

    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if FLAGS.framework == 'tflite':
        pred_bbox = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score,
                                        input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
        pred_bbox = tf.concat([boxes, pred_conf], axis=-1)
    boxes = pred_bbox[:, :, 0:4]
    pred_conf = pred_bbox[:, :, 4:]


    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    #print('boxes:{}'.format(boxes))
    return boxes, scores, classes, valid_detections

def save_tf(weights):
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

  input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
  bbox_tensors = []
  prob_tensors = []
  if FLAGS.tiny:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  else:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      elif i == 1:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)
  if FLAGS.framework == 'tflite':
    pred = (pred_bbox, pred_prob)
  else:
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
  model = tf.keras.Model(input_layer, pred)
  
  #model.save_weights(FLAGS.weights)
  #model.load_weights(FLAGS.weights)
  model.load_weights(weights)
  #utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
  model.summary()
  model.save(FLAGS.output)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def main(_argv):
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #if len(physical_devices) > 0:
        #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    from core.yolov4 import YOLO, decode, compute_loss, decode_train
    trainset = Dataset(FLAGS, is_training=True)
    testset = Dataset(FLAGS, is_training=False)
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)


    optimizer = tf.keras.optimizers.Adam() 
    #optimizer = tf.keras.optimizers.SGD(momentum=0.937)
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    # define training step function
    # @tf.function
    def train_step(image_data, target,epoch):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            #tf.print("=>EPOCH: %4d STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
            #         "prob_loss: %4.2f   total_loss: %4.2f" % (epoch,global_steps, total_steps, optimizer.lr.numpy(),
            #                                                   giou_loss, conf_loss,
            #                                                   prob_loss, total_loss))
            # update learning rate
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                '''
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
                '''
                lr = cfg.TRAIN.LR_INIT -  (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END)*(global_steps/total_steps)
              
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            
            return total_loss,giou_loss,conf_loss,prob_loss,lr
    

    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            #tf.print("=>TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
            #         "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
            #                                                   prob_loss, total_loss))
            return total_loss,giou_loss,conf_loss,prob_loss
    """
     Calculate the AP given the recall and precision array
      1st) We compute a version of the measured precision/recall curve with
           precision monotonically decreasing
      2nd) We compute the AP as the area under this curve by numerical integration.
    """
    def voc_ap(rec, prec):
      """
      --- Official matlab code VOC2012---
      mrec=[0 ; rec ; 1];
      mpre=[0 ; prec ; 0];
      for i=numel(mpre)-1:-1:1
          mpre(i)=max(mpre(i),mpre(i+1));
      end
      i=find(mrec(2:end)~=mrec(1:end-1))+1;
      ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
      """
      rec.insert(0, 0.0) # insert 0.0 at begining of list
      rec.append(1.0) # insert 1.0 at end of list
      mrec = rec[:]
      prec.insert(0, 0.0) # insert 0.0 at begining of list
      prec.append(0.0) # insert 0.0 at end of list
      mpre = prec[:]
      """
       This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab:  for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
      """
      # matlab indexes start in 1 but python in 0, so I have to do:
      #   range(start=(len(mpre) - 2), end=0, step=-1)
      # also the python function range excludes the end, resulting in:
      #   range(start=(len(mpre) - 2), end=-1, step=-1)
      for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
      """
       This part creates a list of indexes where the recall changes
        matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
      """
      i_list = []
      for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
          i_list.append(i) # if it was matlab would be i + 1
      """
       The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
      """
      ap = 0.0
      for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
      return ap, mrec, mpre
    def Validation_mAP():
    #def main(INPUT_SIZE=640):
        tmp_files_path = "tmp_files"
        
        
        if not os.path.exists(tmp_files_path): # if it doesn't exist already
            os.makedirs(tmp_files_path)
        
        INPUT_SIZE = FLAGS.input_size
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)
       
        # Build Model
        if FLAGS.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            #print(input_details)
            #print(output_details)
        else:
            inputs = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
            outputs = YOLO(inputs, NUM_CLASS, FLAGS.model, FLAGS.tiny)
            model = tf.keras.Model(inputs, outputs)
            model.load_weights(FLAGS.output)
            #saved_model_loaded = tf.saved_model.load(weightss, tags=[tag_constants.SERVING])
            #infer = saved_model_loaded.signatures['serving_default']
       
        predict_record = []
        
        gt_counter_per_class = {}
        num_lines = sum(1 for line in open(FLAGS.annotation_path))
        with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
            for num, line in enumerate(tqdm(annotation_file,total=num_lines,ncols=140)):
                ground_truth_record = []
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                 
                
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt = []
                    classes_gt = []
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                
                """
                 Ground-Truth
                   Load each of the ground-truth files into a temporary ".json" file.
                   Create a list of all the class names present in the ground-truth (gt_classes).
                """
                #==========================================================================================================
                num_bbox_gt = len(bboxes_gt)
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    #=======================================================================================
                    bbox = xmin, ymin, xmax, ymax
                    
                    ground_truth_record.append({"file_id": num,  "class_name":class_name, "bbox":bbox, "used":False})
                    
                    if class_name in gt_counter_per_class:
                        gt_counter_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        gt_counter_per_class[class_name] = 1
                    
                    #=======================================================================================
                #file_id = image_path.split(os.sep)[-1]
                #file_id = file_id.split('.')[0]
                # dump bounding_boxes into a ".json" file
                with open(tmp_files_path + "/" + str(num) + "_ground_truth.json", 'w') as outfile:
                    json.dump(ground_truth_record, outfile)
                #==============================================================================================================
               
                '''Predict Process'''
                image_size = image.shape[:2]
                # image_data = utils.image_preprocess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
                image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                #==========================================================================================================
                '''model inference'''
                if FLAGS.framework == 'tflite':
                    interpreter.set_tensor(input_details[0]['index'], image_data)
                    interpreter.invoke()
                    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                
                    if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25)
                    else:
                        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25)
                else:
                    batch_data = tf.constant(image_data)
                    boxes, scores, classes, valid_detections = infer(batch_data, model)
                #============================================================================================================
                '''convert tf to numpy'''
                boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

                # if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
                #     image_result = utils.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
                #     cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image_result)
                #=========================================================================================================
                '''Analysis inference result (after filterbox && nms)'''
                image_h, image_w, _ = image.shape
                for i in range(valid_detections[0]):
                    if int(classes[0][i]) < 0 or int(classes[0][i]) > NUM_CLASS: continue
                    coor = boxes[0][i]
                    coor[0] = int(coor[0] * image_h) #ymin
                    coor[2] = int(coor[2] * image_h) #ymax
                    coor[1] = int(coor[1] * image_w) #xmin
                    coor[3] = int(coor[3] * image_w) #xmax

                    score = scores[0][i]
                    class_ind = int(classes[0][i])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    ymin, xmin, ymax, xmax = list(map(str, coor))
                    bbox =  ymin, xmin, ymax, xmax
                    #=======================================================================================
                    file_id = num
                    #predict_record.append({"confidence":score, "file_id":file_id, "bbox":bbox})
                    predict_record.append([file_id, class_name, score, xmin, ymin, xmax, ymax])
                    #========================================================================================
           
                #print(num, num_lines)
        
       
        gt_classes = list(gt_counter_per_class.keys())
        # let's sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)
        
        
        """
         Predicted
           Load each of the predicted files into a temporary ".json" file.
        """
        for class_index, class_name in enumerate(gt_classes):
            bounding_boxes = []
            for i in range(len(predict_record)):
                image_path, tmp_class_name, confidence, left, top, right, bottom = predict_record[i]
                
                if tmp_class_name == class_name:
                  #print("match")
                  bbox = left + " " + top + " " + right + " " +bottom
                  file_id = image_path
                  bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
              #print(bounding_boxes)
            # sort predictions by decreasing confidence
            bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
            with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
              json.dump(bounding_boxes, outfile)
        
        
        """
         Calculate the AP for each class
        """
        results_files_path = "./mAP/results/"
        sum_AP = 0.0
        ap_dictionary = {}
        # open file to store the results
        MINOVERLAP = 0.5
        final_mpre = 0
        final_mrec = 0
        ap_list = []
        with open(results_files_path + "/results.txt", 'w') as results_file:
            results_file.write("# AP and precision/recall per class\n")
            count_true_positives = {}
            
            for class_index, class_name in enumerate(gt_classes):
                count_true_positives[class_name] = 0
                """
                 Load predictions of that class
                """
                predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
                predictions_data = json.load(open(predictions_file))
                
                """
                 Assign predictions to ground truth objects
                """
                nd = len(predictions_data)
                tp = [0] * nd # creates an array of zeros of size nd
                fp = [0] * nd
                for idx, prediction in enumerate(predictions_data):
                    file_id = prediction["file_id"]
                    # assign prediction to ground truth object if any
                    #   open ground-truth with that file_id
                    gt_file = tmp_files_path + "/" + str(file_id) + "_ground_truth.json"
                    ground_truth_data = json.load(open(gt_file))
                    ovmax = -1
                    gt_match = -1
                    # load prediction bounding-box
                    bb = [ float(x) for x in prediction["bbox"].split() ]
                    for obj in ground_truth_data:
                        # look for a class_name match
                        if obj["class_name"] == class_name:
                            bbgt = [ float(x) for x in obj["bbox"] ]
                            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
                                    gt_match = obj
                    min_overlap = MINOVERLAP
                   
                    if ovmax >= min_overlap:
                        if "difficult" not in gt_match:
                            if not bool(gt_match["used"]):
                                # true positive
                                tp[idx] = 1
                                #print('TP')
                                gt_match["used"] = True
                                count_true_positives[class_name] += 1
                                # update the ".json" file
                                with open(gt_file, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                               
                            else:
                                # false positive (multiple detection)
                                fp[idx] = 1
                                #print('FP')
                    else:
                        # false positive
                        fp[idx] = 1
                        #print('FP')
                        if ovmax > 0:
                            status = "INSUFFICIENT OVERLAP"
                #print(tp)
                '''compute precision/recall of each class'''
                cumsum = 0
                for idx, val in enumerate(fp):
                  fp[idx] += cumsum
                  cumsum += val
                cumsum = 0
                for idx, val in enumerate(tp):
                  tp[idx] += cumsum
                  cumsum += val
                #print(tp)
                rec = tp[:]
                for idx, val in enumerate(tp):
                  rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
                #print(rec)
                prec = tp[:]
                for idx, val in enumerate(tp):
                  prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
                #print(prec)
                
                if len(rec)>0:  
                    #m_mrec = float(sum(rec)/len(rec))
                    m_mrec = float(rec[len(rec)-1])
                else:
                    m_mrec=0.0
                if len(prec)>0:
                    #m_mprec = float(sum(prec)/len(prec))
                    m_mprec = float(prec[len(prec)-1])
                else:
                    m_mprec = 0.0
                
                final_mpre+=m_mprec
                final_mrec+=m_mrec
                '''calculate AP of each class'''
                ap, mrec, mprec = voc_ap(rec, prec)
                sum_AP += ap
                ap_list.append([class_name,ap])
                text = "{0:.3f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)
                """
                 Write to results.txt
                """
                rounded_prec = [ '%.2f' % elem for elem in prec ]
                rounded_rec = [ '%.2f' % elem for elem in rec ]
                results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
                
                print(text)
                ap_dictionary[class_name] = ap
            '''End for class_index, class_name in enumerate(gt_classes):'''
        '''End with open(results_files_path + "/results.txt", 'w') as results_file:'''
        
        '''calcualte mAP, mean_pre, mean_rec'''
        mAP = sum_AP/n_classes
        final_mrec = final_mrec/n_classes
        final_mpre = final_mpre/n_classes
        # remove the tmp_files directory
        #if os.path.exists(tmp_files_path):
        #shutil.rmtree(tmp_files_path)
        return mAP,final_mrec,final_mpre
        
   #=======================End of Validation_mAP()===================================================================================
                
    total_epoch = first_stage_epochs + second_stage_epochs
    VAL_LOSS = 100000
    for epoch in range(first_stage_epochs + second_stage_epochs):
        records = []
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        
        DO_TRAIN=True
        if DO_TRAIN:
            print("")
            print("---------------------------------------------------------------------------------------------")
            print('{}{:8}{}{:8}{}{:8}{}{:8}{}{:8}{}{:8}{}'.format('Epoch','','Total_loss','','box','','obj','','cls','','img_size','','lr'))
            #print(' Epoch        Total_loss      box         obj          cls      img_size')
            #print('====================================================================================')
            pbar_train = tqdm(trainset,ncols=140)
            pbar_test =  tqdm(testset,ncols=140)
            Total_Train_Loss, Total_giou_loss, Total_conf_loss, Total_prob_loss = 0,0,0,0
            for image_data, target in pbar_train:
                total_loss_train,giou_loss,conf_loss,prob_loss,lr = train_step(image_data, target, epoch)
                
                
                bar_str =  '  ' +  str(epoch+1) + '/' + str(total_epoch)\
                          + '{:8}'.format('')+"{0:.3f}".format(total_loss_train)\
                          + '{:8}'.format('')+"{0:.3f}".format(giou_loss)\
                          + '{:8}'.format('')+"{0:.3f}".format(conf_loss)\
                          + '{:8}'.format('')+"{0:.3f}".format(prob_loss)\
                          + '{:8}'.format('')+"{0:.3f}".format(cfg.TRAIN.INPUT_SIZE)\
                          + '{:8}'.format('')+"{0:.6f}".format(lr)
                PREFIX = colorstr(bar_str)
                pbar_train.desc = f'{PREFIX}'
                
                Total_Train_Loss+=total_loss_train
                Total_giou_loss+=giou_loss
                Total_conf_loss+=conf_loss
                Total_prob_loss+=prob_loss
                
            records.append(['Train', epoch+1, Total_Train_Loss.numpy(), Total_giou_loss.numpy(), Total_conf_loss.numpy(), Total_prob_loss.numpy(), lr])
        DO_VAL = True
        save_valloss_min_model = True
        Total_Val_Loss, Total_val_giou_loss, Total_val_conf_loss, Total_val_prob_loss = 0,0,0,0
        if DO_VAL and (epoch+1)>=4:
            
            print('{}{:10}{}{:10}{}{:10}{}'.format('Total_loss','','box','','obj','','cls'))
            print('     ------------------------------------------------------------')
            save_valloss_min_model = False
            for image_data, target in pbar_test:
                total_loss_val,giou_loss,conf_loss,prob_loss = test_step(image_data, target)
                
               
                
                bar_str =   '         '+ "{0:.3f}".format(total_loss_val)\
                          + '         ' + "{0:.3f}".format(giou_loss)\
                          + '      ' + "{0:.3f}".format(conf_loss)\
                          + '      ' + "{0:.3f}".format(prob_loss)
                          
                PREFIX = colorstr(bar_str)
                pbar_test.desc = f'{PREFIX}'
                #pbar_test.update(1)
                Total_Val_Loss+=total_loss_train
                Total_val_giou_loss+=giou_loss
                Total_val_conf_loss+=conf_loss
                Total_val_prob_loss+=prob_loss
                
            
            if Total_Val_Loss < VAL_LOSS:
                VAL_LOSS = Total_Val_Loss
                save_valloss_min_model = True
            #records.append(['Val  ', epoch+1, Total_Val_Loss.numpy(), Total_val_giou_loss.numpy(), Total_val_conf_loss.numpy(), Total_val_prob_loss.numpy()])
        if save_valloss_min_model:
        #if True:
            #print('Best Val loss: {} , Total_Val_Loss : {} start to save best and current model'.format(VAL_LOSS,Total_Val_Loss))
            #tf.saved_model.save(model, './model')
            model.save_weights(FLAGS.output_best)
            model.save_weights(FLAGS.output)
            #save_tf(weights='./checkpoints_yolov4_20220729_ciou_tf25_mosaic_aug_test/yolov4')    
            #model.save('./model_20220731')
        else:
            #print('Best Val loss: {} , Total_Val_Loss : {} start to save current model'.format(VAL_LOSS,Total_Val_Loss))
            model.save_weights(FLAGS.output)
        
        if (epoch+1)>=10:
            mAP, m_mrec, m_mprec = Validation_mAP()
            output = './mAP/results'
            mAP_text = "{0:.3f}".format(mAP)
            m_mrec_text = "{0:.3f}".format(m_mrec)
            m_mprec_text = "{0:.3f}".format(m_mprec)
            
            #print(ap_list)
            full_text = '        ' + m_mprec_text + '        ' + m_mrec_text + '        ' + mAP_text
            PREFIX = colorstr(full_text)
            column    = '         P             R           mAP@.5   '
            records.append(['Val  ', epoch+1, Total_Val_Loss.numpy(), Total_val_giou_loss.numpy(), Total_val_conf_loss.numpy(), Total_val_prob_loss.numpy(), m_mprec, m_mrec, mAP])
            print(column)
            print('    ----------------------------------------------')
            print(PREFIX)
        import csv
        result_path = './models/checkpoints_yolov4_20220813/result.txt'
        result_dir = './models/checkpoints_yolov4_20220813'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        fields = ['data', 'Epoch', 'Total_loss', 'giou_loss', 'conf_loss', 'prob_loss', 'precision', 'recall', 'mAP@0.5' ]
        
        with open(result_path, 'a') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(records)
            
        
     
            # using csv.writer method from CSV package
            
    '''    
    import csv
    result_path = './train/checkpoints_yolov4_20220812.csv'
    result_dir = './train'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    fields = ['data', 'Epoch', 'Total_loss', 'giou_loss', 'conf_loss', 'prob_loss', 'mAP_0.5']
    with open(result_path, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(records)
    '''    
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass