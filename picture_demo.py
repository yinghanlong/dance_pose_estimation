import os
import re
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)



model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
# model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()
start_time = time.perf_counter_ns()
test_image = '/content/drive/MyDrive/pytorch_Realtime_Multi-Person_Pose_Estimation/pose.png'
oriImg = cv2.imread(test_image) # B,G,R order
shape_dst = np.min(oriImg.shape[0:2])
# Get results of original image

with torch.no_grad():
    paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')
          
# print(im_scale)
humans = paf_to_pose_cpp(heatmap, paf, cfg)
elapsed_time = ( time.perf_counter_ns() - start_time ) / 1000000000

print("took {} seconds".format(elapsed_time))   
out = draw_humans(oriImg, humans)

image_h, image_w = oriImg.shape[:2]
bounding_boxes = []
for human in humans:
  bounding_box = human.get_upper_body_box(image_w, image_h) #
  bounding_boxes.append(bounding_box)
  # for i in human.body_parts.keys():
  #   print (i, " : " , "x: ", human.body_parts[i].x, "y: ", human.body_parts[i].y) 0-17

cv2.imwrite('result.png',out)   


