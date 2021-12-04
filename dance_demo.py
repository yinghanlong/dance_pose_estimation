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
import copy
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
    #print(H)
    return H
    
def hconcat(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = im_list[0].shape[0]
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
      
    concat_img=cv2.hconcat(im_list_resize)
    #print(concat_img.shape)
    return concat_img

def map_keypoints(keypoints,h, w, H=None):
    #map the points
    if H is not None:
      Hinv = np.linalg.inv(H)
    mapped_keypoints= np.zeros((18,2))
    cnt=0
    for i in keypoints.keys():
        col= int(keypoints[i].x * w +0.5) #x
        row= int(keypoints[i].y * h +0.5) #y
        x= np.transpose(np.array([col,row,1]))
        if H is not None:
          xproj = np.dot(Hinv, x)
          xproj = xproj/xproj[2]
        else:
          xproj = x
        
        if cnt<=17:
          mapped_keypoints[cnt,0]= xproj[0]
          mapped_keypoints[cnt,1]= xproj[1]
        cnt+=1 
    return mapped_keypoints

parser = argparse.ArgumentParser(description='DancePose Demo')

parser.add_argument("--studentvideo", type=str,help='video path', default='student.mp4')
parser.add_argument("--teachervideo", type=str,help='video path', default='teacher.mp4')
parser.add_argument('--num_player', type=int,
                    default=1)
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
    video_path = '/content/drive/MyDrive/pytorch_Realtime_Multi-Person_Pose_Estimation/'
    video_path1 = video_path+ args.studentvideo
    print(video_path1)
    video_capture = cv2.VideoCapture(video_path1)
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))

    video_test_path = video_path + args.teachervideo
    print(video_test_path)
    video_capture2 = cv2.VideoCapture(video_test_path)
    frame_width_2 = int(video_capture2.get(3))
    frame_height_2 = int(video_capture2.get(4))
    out_video = cv2.VideoWriter('out_'+args.studentvideo,cv2.VideoWriter_fourcc('m','p','4','v'), 20, (int(frame_width+frame_width_2*frame_height/frame_height_2),frame_height))

    count = 0
    score_cnt=0
    # print(cv2.CAP_PROP_FRAME_HEIGHT)
    player_scores= np.zeros(args.num_player)
    #synchronize video1&2
    #for i in range(52):
    #    ret, oriImg = video_capture.read()
    fps1 =  int(video_capture.get(cv2.CAP_PROP_FPS))
    fps2 =  int(video_capture2.get(cv2.CAP_PROP_FPS))
    MIN_FPS=int(min(fps1,fps2))
    print('Frames per second',fps1,fps2)
    while True:
        # Capture frame-by-frame
        # video_capture.set(cv2.CAP_PROP_POS_MSEC,(count * 10000))
        count +=1
        ret, oriImg = video_capture.read()
        ret2, oriImg2 = video_capture2.read()

        if ret == True and ret2 == True:
          shape_dst = np.min(oriImg.shape[0:2])
          shape_dst_2 = np.min(oriImg2.shape[0:2])
          if count % MIN_FPS == 0:
            
            #deal with different frame rates
            if fps1>fps2:
              for f in range(int(fps1-MIN_FPS)):
                ret, oriImg = video_capture.read()
            elif fps1<fps2:
              for f in range(int(fps2-MIN_FPS)):
                ret2, oriImg2 = video_capture2.read()
            if ret is not True or ret2 is not True:
              break
            with torch.no_grad():
                paf, heatmap, imscale = get_outputs(
                    oriImg, model, 'rtpose')
                paf2, heatmap2, imscale2 = get_outputs(
                    oriImg2, model, 'rtpose')   
            humans = paf_to_pose_cpp(heatmap, paf, cfg)
            print('Detected number of players:',len(humans))
            humans2 = paf_to_pose_cpp(heatmap2, paf2, cfg)
            out = draw_humans(oriImg, humans, imgcopy=True)
            teacher_out = draw_humans(oriImg2, humans2)
            image_h, image_w = oriImg.shape[:2]
            image_h_2, image_w_2 = oriImg2.shape[:2]
            bounding_boxes = []
            bounding_boxes_2 = []
            for human in humans:
              bounding_box = human.get_upper_body_box(image_w, image_h) #
              if bounding_box != None:
                bounding_boxes.append(bounding_box)
            for human in humans2:
              bounding_box = human.get_upper_body_box(image_w_2, image_h_2) #
              if bounding_box!= None:
                bounding_boxes_2.append(bounding_box)
              # for i in human.body_parts.keys():
              #   print (i, " : " , "x: ", human.body_parts[i].x, "y: ", human.body_parts[i].y) 0-17
            if bounding_boxes == None or len(bounding_boxes) == 0 or bounding_boxes_2 == None or len(bounding_boxes_2) == 0:
              concat_img=hconcat([oriImg,oriImg2])
              out_video.write(concat_img)
              continue
            #Teacher's bbox location
            teacher_body= humans2[0].body_parts
            b= np.zeros((8))
            tbox_x= bounding_boxes_2[0]["x"]
            tbox_y= bounding_boxes_2[0]["y"]
            tbox_w= bounding_boxes_2[0]["w"]
            tbox_h= bounding_boxes_2[0]["h"]
            b= np.array([max(0,tbox_x- tbox_w/2), max(0,tbox_y- tbox_h/2),min(image_w_2,tbox_x+ tbox_w/2), max(0,tbox_y- tbox_h/2),max(0,tbox_x- tbox_w/2),min(image_h_2, tbox_y+tbox_h/2),min(image_w_2,tbox_x+ tbox_w/2),min(image_h_2, tbox_y+tbox_h/2)])

            #Draw bounding box
            #cv2.rectangle(oriImg2, (int(b[0]), int(b[1])), (int(b[6]), int(b[7])), (255,0,0), 2)
            #if bounding boxes sizes are very different, skip
            #if tbox_h> 1.5* bounding_boxes[0]["h"] or tbox_h<0.7*bounding_boxes[0]["h"] or tbox_w> 1.5* bounding_boxes[0]["w"] or tbox_w<0.7*bounding_boxes[0]["w"]:
            #  concat_img=hconcat([oriImg,oriImg2])
            #  out_video.write(concat_img)
            #  continue
           
            #compute scores for multiple players
            teacher_pose = out
            score_cnt+=1
            for player in range(args.num_player):
              if player>= len(bounding_boxes):
                break
    
              pbox_x= bounding_boxes[player]["x"]
              pbox_y= bounding_boxes[player]["y"]
              pbox_w= bounding_boxes[player]["w"]
              pbox_h= bounding_boxes[player]["h"]
              P= np.array([max(0,pbox_x- pbox_w/2), max(0,pbox_y- pbox_h/2),1])
              Q= np.array([min(image_w,pbox_x+ pbox_w/2), max(0,pbox_y- pbox_h/2),1])
              R= np.array([max(0,pbox_x- pbox_w/2),min(image_h, pbox_y+pbox_h/2),1])
              S= np.array([min(image_w,pbox_x+ pbox_w/2),min(image_h, pbox_y+pbox_h/2),1])
              #Draw bounding box
              #cv2.rectangle(teacher_pose, (int(P[0]), int(P[1])), (int(S[0]), int(S[1])), (255,0,0), 2)
              H= homography(P,Q,R,S, b)
              
              mapped_keypoints1 = map_keypoints(humans[player].body_parts,image_h,image_w)
              mapped_keypoints2 = map_keypoints(teacher_body,image_h_2,image_w_2,H)

              distance = compare(mapped_keypoints1, mapped_keypoints2)
              score = -0.5*distance +105
              score = min(100,max(0,score)) #make sure score is between 0 and 100
              player_scores[player]+=score
              print('frame ', count, ', distance=',distance, ', score=',score)
              #print(mapped_keypoints1,mapped_keypoints2)
              
              teacher_points=[]
              teacher_points.append(copy.deepcopy(humans2[0]))
              cnt17=0
              for k in teacher_points[0].body_parts.keys():
                teacher_points[0].body_parts[k].x= mapped_keypoints2[cnt17,0]/image_w
                teacher_points[0].body_parts[k].y= mapped_keypoints2[cnt17,1]/image_h
                if cnt17==17:
                  break
                cnt17+=1
              
              teacher_pose = draw_humans(teacher_pose, teacher_points, imgcopy=True, color=False)
              font = cv2.FONT_HERSHEY_COMPLEX
              print_score="SCORE: "+str(int(score))
              if score>95:
                print_score= 'Perfect! '+print_score
              elif score>80:
                print_score= 'Excellent! '+print_score
              elif score>60:
                print_score= 'Good! '+print_score
              else:
                print_score= 'Miss. '+print_score
              cv2.putText(teacher_pose, 'Player '+str(int(player)), (max(10,pbox_x-50),50),font,0.7,(255,255,255),2)
              cv2.putText(teacher_pose, print_score, (max(10,pbox_x-50),100),font,0.7,(255,255,255),2)  #text,coordinate,font,size of text,color,thickness of font
              #if score <80:
                #cv2.imwrite("./dance_results/wrong_pose_"+str(count)+"_"+str(player)+".png",hconcat([teacher_pose,oriImg2]))
              #if score>95:
                #correct_pose = draw_humans(oriImg, teacher_points, imgcopy=True, color=False)
                #cv2.imwrite("./dance_results/correct_pose_"+str(count)+"_"+str(player)+".png",hconcat([teacher_pose,oriImg2]))
            for j in range(5):
              concat_img=hconcat([out,oriImg2])
              out_video.write(concat_img)
            for j in range(10):
              concat_img=hconcat([teacher_pose,oriImg2])
              out_video.write(concat_img)
            #only process first N frames
            if count>3000:
              break
          else:
            concat_img=hconcat([oriImg,oriImg2])
            out_video.write(concat_img)
          # Display the resulting frame
          #cv2.imwrite('Video', out)

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
        else:
          break
    player_scores /= score_cnt
    print('Total score=', player_scores)
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
