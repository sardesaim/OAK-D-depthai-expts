# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:45:14 2020

@author: sarde
"""
from tqdm import tqdm
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import imageio
import glob
import ntpath
import time
import os
from tensorflow.keras import backend as K
import argparse
from pathlib import Path
from multiprocessing import Process
from time import time
try:
    from armv7l.openvino.inference_engine import IENetwork, IECore
except:
    from openvino.inference_engine import IENetwork, IECore

from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=16)

parser = argparse.ArgumentParser()
parser.add_argument('-ip', '--input_path', \
    default='D:/00_NCSU/00_Resources/00_Datasets/oak_NC_MD_grassclover/val/', \
    type=str, help="Input Path")
parser.add_argument("--xml_path",\
                    default="/home/pi/OAK-D-depthai-expts/02-NCS2-mode/FP16/3class_360/3class_360.xml", \
                        help="Path of the deeplabv3plus openvino model.")
parser.add_argument('-pb_path', '--tf_pb_path', \
    default = "D:/00_NCSU/Fall2020/ECE633_IndividualTopics/OAK-D-Weed-Cam/Model/deeplabv3+/models/3_class_model_mobilenet_v3_small_v2.1/3_class_model_mobilenet_v3_small_v2.1_1080x1920.pb", \
        type=str, help='Model Path for tensorflow file')
parser.add_argument('-ms', '--model_size', default = (1080,1920), type=int, help='Model Input size')
parser.add_argument('--device', type=str, default='MYRIAD', help='Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
                                                                   Sample will look for a suitable plugin for device specified (CPU by default)')
args = parser.parse_args()

images_path= args.input_path + 'images/'
labels_path= args.input_path + 'labels/'
current_input_size=args.model_size
current_model = args.model_path
model_xml = args.deep_model
model_bin = os.path.splitext(model_xml)[0] + ".bin"

class DeepLabModel():
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    SOFTMAX_TENSOR_NAME = 'SemanticProbabilities:0'
    INPUT_SIZE = current_input_size


    def __init__(self, path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.

        with tf.gfile.GFile(path, 'rb')as file_handle:
            graph_def = tf.GraphDef.FromString(file_handle.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        
        # To Run on CPU, uncomment below and add config to self.session as: ", config=config"
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
            )

        self.sess = tf.Session(graph=self.graph, config=config)


    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          seg_map: np.array. values of pixels are classes
        """

        # width, height,ch = image.shape

        # resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        # target_size = (int(resize_ratio * width), int(resize_ratio * height))

        # resized_image = cv2.resize(image, (target_size))

        batch_seg_map, batch_prob_map = self.sess.run(
           [self.OUTPUT_TENSOR_NAME,
           self.SOFTMAX_TENSOR_NAME],
           feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]})

        seg_map = batch_seg_map[0]        
        # seg_map = resize(seg_map.astype(np.uint8), (height, width), preserve_range=True, order=0, anti_aliasing=False)
        prob_map = batch_prob_map[0]        
        return seg_map, prob_map

class SegMetrics:
    def __init__(self):
        pass
    def pixel_accuracy(self, groundtruth, preds):
        return np.sum(groundtruth==preds)/(1920*1080)
    
    def IoU(self,groundtruth, preds):
        intersection = np.logical_and(groundtruth, preds)
        union = np.logical_or(groundtruth, preds)
        iou_score = np.sum(intersection) / np.sum(union)
        # print(‘IoU is %s’ % iou_score)
        return iou_score


def image_resize(image):
    model_input = current_input_size
    width_scale = model_input[1] / image.shape[1]
    height_scale = model_input[0] / image.shape[0]
    scale = max((width_scale, height_scale))
    
    model_ip = cv2.resize(image, (int(np.ceil(image.shape[1]*scale)), int(np.ceil(image.shape[0]*scale))))
    model_ip = model_ip[0:model_input[0], 0:model_input[1]]
    # model_up= cv2.resize(model_ip, (1920,1080))
    return model_ip

def load_data(images_path, labels_path):
    data=[]
    labels=[]
    for ims,labs in tqdm(zip(os.listdir(images_path),os.listdir(labels_path))):
        # print(root+dirs+'/'+image)
        img=cv2.imread(images_path+ims)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img=cv2.resize(img, (240,240))
        img = image_resize(img)
        # print(img.shape)
        lab=cv2.imread(labels_path+labs)
        lab[lab==3]=1
        lab[lab==4]=1
        lab[lab==5]=2
        lab[lab==6]=2
        lab = image_resize(lab)
        lab =lab[:,:,0]
        # print(lab.shape)
        data.append(img)
        labels.append(lab)
    return data, labels

data, labels = load_data(images_path, labels_path)
data, labels = np.array(data), np.array(labels)

deeplab_tf = DeepLabModel(current_model)
ie = IECore()
net = ie.read_network(model_xml, model_bin)
input_info = net.input_info
input_blob = next(iter(input_info))
exec_net = ie.load_network(network=net, device_name=args.device)


segs_tf = []
segs_openvino=[]
pixel_accs_tf=[]
ious_tf=[]

pixel_accs_openvino=[]
ious_openvino=[]
seg_met=SegMetrics()
for i in tqdm(range(data.shape[0])):
    prepimg_deep = np.expand_dims(data[i], axis=0)
    prepimg_deep = prepimg_deep.astype(np.float32)
    prepimg_deep = np.transpose(prepimg_deep, [0, 3, 1, 2])
    t1 = time.perf_counter()
    seg, _=deeplab_tf.run(data[i])
    t2 = time.perf_counter()
    deeplabv3_predictions = exec_net.infer(inputs={input_blob: prepimg_deep})
    t2 = time.perf_counter()
    #append preds to their lists
    segs_tf.append(seg)
    segs_openvino.append(deeplabv3_predictions)
    #calculate and append pixel accuracies/ IoU  tensorflow. 
    pixel_accs_tf.append(seg_met.pixel_accuracy(labels[i],segs_tf[i]))
    ious_tf.append(seg_met.IoU(labels[i],segs_tf[i]))
    #calculate and append pixel accuracies/ IoU openvino
    pixel_accs_openvino.append(seg_met.pixel_accuracy(labels[i],segs_openvino[i]))
    ious_openvino.append(seg_met.IoU(labels[i],segs_openvino[i]))
#print mean pixel accuracy and IoU 
print('Mean pixel accuracy :', np.mean(pixel_accs))
print('Mean IoUs:', np.mean(ious))
# print(seg_met.IoU(labels[0],segs[0]))
