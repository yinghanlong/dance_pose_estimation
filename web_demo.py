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
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from lib.config import update_config, cfg
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp


def compare(pose1,pose2):
    diff = np.mean(abs(pose1-pose2))
    return diff

def homography(P,Q,R,S,b):
    A= np.zeros((8,8))
    A[0,0:3]=P
    A[1,3:6]=P
    A[2,0:3]=Q
    A[3,3:6]=Q
    A[4,0:3]=R
    A[5,3:6]=R
    A[6,0:3]=S
    A[7,3:6]=S
    for j in range(0,4):
        A[2*j,6:8]= -b[2*j] * A[2*j,0:2]
        A[2*j+1,6:8]= -b[2*j+1] * A[2*j+1,3:5]
    #print(A)
    #Calculate the homography        
    h= np.dot(np.linalg.inv(A),np.transpose(b))

    H= np.zeros((3,3))
    H[0,:]= h[0:3]
    H[1,:]= h[3:6]
    H[2,0:2]= h[6:9]
    H[2,2]=1
    print(H)
    return H
    
def map_figs(imgfill,img, paint, H):
    #map the points
    for col in range(0,imgfill.shape[1]):
        for row in range(0,imgfill.shape[0]):
            x= np.transpose(np.array([col,row,1]))
            if (imgfill[row,col,1]>0):
                Hinv = np.linalg.inv(H)
                xproj = np.dot(Hinv, x)
                xproj = xproj/xproj[2]
                rowint =int(xproj[1])
                colint =int(xproj[0])
                img[row,col,:]= paint[rowint,colint,:]

    return img

def map_keypoints(keypoints, H=None):
    #map the points
    if H is not None:
      Hinv = np.linalg.inv(H)
    mapped_keypoints= np.zeros((17,2))
    cnt=0
    for i in keypoints.keys():
        col= keypoints[i].x #x
        row= keypoints[i].y #y
        x= np.transpose(np.array([col,row,1]))
        if H is not None:
          xproj = np.dot(Hinv, x)
          xproj = xproj/xproj[2]
          rowint =int(xproj[1])
          colint =int(xproj[0])
        else:
          rowint = int(x[1])
          colint = int(x[0])
        
        if cnt<17:
          mapped_keypoints[cnt,0]= colint
          mapped_keypoints[cnt,1]= rowint 
        cnt+=1 
    return mapped_keypoints
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

model.float()
model.eval()

if __name__ == "__main__":

    video_path = "/content/drive/MyDrive/pytorch_Realtime_Multi-Person_Pose_Estimation/student.mp4"
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    out_video = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    video_test_path =  "/content/drive/MyDrive/pytorch_Realtime_Multi-Person_Pose_Estimation/teacher.mp4"
    video_capture2 = cv2.VideoCapture(video_test_path)
    frame_width_2 = int(video_capture2.get(3))
    frame_height_2 = int(video_capture2.get(4))

    count = 0
    # print(cv2.CAP_PROP_FRAME_HEIGHT)
    while True:
        # Capture frame-by-frame
        # video_capture.set(cv2.CAP_PROP_POS_MSEC,(count * 10000))
        count +=1
        ret, oriImg = video_capture.read()
        ret2, oriImg2 = video_capture2.read()
        if ret == True and ret2 == True:
          shape_dst = np.min(oriImg.shape[0:2])
          shape_dst_2 = np.min(oriImg2.shape[0:2])
          if count % 50 == 0:
            with torch.no_grad():
                paf, heatmap, imscale = get_outputs(
                    oriImg, model, 'rtpose')
                paf2, heatmap2, imscale2 = get_outputs(
                    oriImg2, model, 'rtpose')   
            humans = paf_to_pose_cpp(heatmap, paf, cfg)
            humans2 = paf_to_pose_cpp(heatmap2, paf2, cfg)
            out = draw_humans(oriImg, humans)
            image_h, image_w = oriImg.shape[:2]
            bounding_boxes = []
            bounding_boxes_2 = []
            for human in humans:
              bounding_box = human.get_upper_body_box(image_w, image_h) #
              if bounding_box != None:
                bounding_boxes.append(bounding_box)
            for human in humans2:
              bounding_box = human.get_upper_body_box(image_w, image_h) #
              if bounding_boxes_2!= None:
                bounding_boxes_2.append(bounding_box)
              # for i in human.body_parts.keys():
              #   print (i, " : " , "x: ", human.body_parts[i].x, "y: ", human.body_parts[i].y) 0-17
            if bounding_boxes == None or len(bounding_boxes) == 0:
              out_video.write(oriImg)
              continue
            pbox_x= bounding_boxes[0]["x"]
            pbox_y= bounding_boxes[0]["y"]
            pbox_w= bounding_boxes[0]["w"]
            pbox_h= bounding_boxes[0]["h"]
            P= np.array([max(0,pbox_x- pbox_w/2), max(0,pbox_y- pbox_h/2),1])
            Q= np.array([min(image_w,pbox_x+ pbox_w/2), max(0,pbox_y- pbox_h/2),1])
            R= np.array([max(0,pbox_x- pbox_w/2),min(image_h, pbox_y+pbox_h/2),1])
            S= np.array([min(image_w,pbox_x+ pbox_w/2),min(image_h, pbox_y+pbox_h/2),1])
            #Teacher's bbox location
            b= np.zeros((8))
            tbox_x= bounding_boxes_2[0]["x"]
            tbox_y= bounding_boxes_2[0]["y"]
            tbox_w= bounding_boxes_2[0]["w"]
            tbox_h= bounding_boxes_2[0]["h"]
            b= np.array([max(0,tbox_x- tbox_w/2), max(0,tbox_y- tbox_h/2),min(image_w,tbox_x+ tbox_w/2), max(0,tbox_y- tbox_h/2),max(0,tbox_x- tbox_w/2),min(image_h, tbox_y+tbox_h/2),min(image_w,tbox_x+ tbox_w/2),min(image_h, tbox_y+tbox_h/2)])

            H= homography(P,Q,R,S, b)

            mapped_keypoints1 = map_keypoints(humans[0].body_parts)
            mapped_keypoints2 = map_keypoints(humans[0].body_parts,H)
            score= compare(mapped_keypoints1, mapped_keypoints2)
            print('frame ', count, ', distance=',score)
            if score > 80:
              cv2.imwrite("student_l.png",oriImg)
              cv2.imwrite("teacher_l.png",oriImg2)
            if score < 10:
              cv2.imwrite("student_s.png",oriImg)
              cv2.imwrite("teacher_s.png",oriImg2)
            out_video.write(out)
            out_video.write(out)
          else:
            out_video.write(oriImg)
          # Display the resulting frame
          #cv2.imwrite('Video', out)

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
        else:
          break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
