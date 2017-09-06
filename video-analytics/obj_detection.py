import cv2
import tensorflow as tf
import numpy as np
import matplotlib 
import sys
import os
import time

sys.path.append('../../models')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util 

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'

PATH_TO_LABELS = os.path.join('../../models', 'object_detection', 'data', 'mscoco_label_map.pbtxt')
PATH_TO_CKPT = os.path.join('..', MODEL_NAME, 'frozen_inference_graph.pb')
NUM_CLASSES = 90


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def LoadTfModel():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)
        return detection_graph, sess

def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    time1 = time.time()
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    time2 = time.time()
    print "pre inference time: %f" % (time2 - time1)
    # Actual detection.
    time1 = time.time()
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    time2 = time.time()
    print "actual inference time: %f" % (time2 - time1)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np

if __name__ == '__main__':
    time1 = time.time()
    tf_graph, sess = LoadTfModel()
    time2 = time.time()
    print "loading tf model: %f" % (time2 - time1)
    cap = cv2.VideoCapture(0)

    frame_count = 0;
    print('starting video processing...')
    while True:
        time1 = time.time()
        ret, frame = cap.read()
        time2 = time.time()
        print "frame capture time: %f" % (time2 - time1)

        time1 = time.time()
        frame_resized = cv2.resize(frame, (300, 300))
        time2 = time.time()
        print "frame resize time: %f" % (time2 - time1)

        time1 = time.time()
        inferred_frame = detect_objects(frame_resized, sess, tf_graph)
        time2 = time.time()
        print "inference time: %f" % (time2 - time1)

        cv2.imshow('output', inferred_frame)
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

        if frame_count % 30 == 0:
            print 'Frame count2: %d' % (frame_count)

        frame_count += 1

        

    sess.close()
    cap.release()
    cv2.destroyAllWindows()