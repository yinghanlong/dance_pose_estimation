import numpy as np
import pickle
import cv2
import os
_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
           [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
           [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
           [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
           [255, 0, 170], [255, 0, 85]]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

def map_keypoints(keypoints, H):
    #map the points
    Hinv = np.linalg.inv(H)
    mapped_keypoints=keypoints
    for i in range(keypoints.shape[0]):
        col= keypoints[i][0]#x
        row= keypoints[i][1]#y
        x= np.transpose(np.array([col,row,1]))
        xproj = np.dot(Hinv, x)
        xproj = xproj/xproj[2]
        rowint =int(xproj[1])
        colint =int(xproj[0])
        mapped_keypoints[i][0]= colint
        mapped_keypoints[i][1]= rowint 
    return mapped_keypoints

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

def load_keypoint2d(keypoint_dir, seq_name):
    """Load a 2D keypoint sequence represented using COCO format."""
    file_path = os.path.join(keypoint_dir, f'{seq_name}.pkl')
    assert os.path.exists(file_path), f'File {file_path} does not exist!'
    with open(file_path, 'rb') as f:
      data = pickle.load(f)
    if 'det_scores' in data:
      keypoints2d = data['keypoints2d']  # (nviews, N, 17, 3)
      det_scores = data['det_scores']  # (nviews, N)
      timestamps = data['timestamps']  # (N,)
      return keypoints2d, det_scores, timestamps
    else:
      keypoints2d = data['keypoints2d']  # (nviews, nframes, (nsubjects, (133, 3)))
      bboxes = data['bboxes']  # (nviews, (nframes, (nsubjects, (5,))))
      timestamps = data['timestamps']  # (nviews, (nframes,))
      return keypoints2d, bboxes, timestamps

def plot_kpt(keypoint, canvas, color=None):
  for i, (x, y) in enumerate(keypoint[:, 0:2]):
    if np.isnan(x) or np.isnan(y):
      continue
    cv2.circle(canvas, (int(x), int(y)),
               7,
               color if color is not None else _COLORS[i % len(_COLORS)],
               thickness=-1)
  return canvas

#TODO
def compare(pose1,pose2):
    diff = np.mean(abs(pose1-pose2))
    return diff

def scale_and_normalize(pose1,pose2): 
    m1= np.mean(pose1)
    m2= np.mean(pose2)
    norm_pose1=  pose1- m1
    scale1= np.mean(norm_pose1)
    norm_pose1=norm_pose1/scale1
    norm_pose2= pose2- m2
    scale2= np.mean(norm_pose2)
    norm_pose2=norm_pose2/scale2
    return norm_pose1,norm_pose2


filename1='gLO_sBM_cAll_d13_mLO0_ch01'
filename2='gLO_sBM_cAll_d13_mLO0_ch01'
keypoint_dir= 'C:/Users/Yinghan/Documents/academic_2021/keypoints2d'
pose1, _, _= load_keypoint2d(keypoint_dir, seq_name=filename1)
view_idx=1

pose1 =pose1[view_idx, :, :, 0:2]  #[view, n_frames, 17, xy]
pose2, _, _= load_keypoint2d(keypoint_dir, seq_name=filename2)
pose2 =pose2[view_idx, :, :, 0:2]  
total_score = 0

img_1 = np.zeros([1920,1080,1],dtype=np.uint8)
img_1 = plot_kpt(pose1[0,:,:], img_1)
cv2.imwrite('./pose_keypoints.png',img_1)
n_frames1= len(pose1[:,0,0])
n_frames2= len(pose2[:,0,0])
n_frames=min(n_frames1,n_frames2)
print('number of frames to compare',n_frames/30)

#TODO: get bounding boxes
#Player's bbox location
img1=cv2.imread('1.jpg')
paint=cv2.imread('2.jpg')
image_w= img1.shape[0]
image_h= img1.shape[1]
pbox_x=
pbox_y=
pbox_w=
pbox_h=
P= np.array([max(0,pbox_x- pbox_w/2), max(0,pbox_y- pbox_h/2),1])
Q= np.array([min(image_w,pbox_x+ pbox_w/2), max(0,pbox_y- pbox_h/2),1])
R= np.array([max(0,pbox_x- pbox_w/2),min(image_h, pbox_y+pbox_h/2),1])
S= np.array([min(image_w,pbox_x+ pbox_w/2),min(image_h, pbox_y+pbox_h/2),1])
#Teacher's bbox location
b= np.zeros((8))
tbox_x=
tbox_y=
tbox_w=
tbox_h=
b= np.array([max(0,tbox_x- tbox_w/2), max(0,tbox_y- tbox_h/2),min(image_w,tbox_x+ tbox_w/2), max(0,tbox_y- tbox_h/2),max(0,tbox_x- tbox_w/2),min(image_h, tbox_y+tbox_h/2),min(image_w,tbox_x+ tbox_w/2),min(image_h, tbox_y+tbox_h/2)])

H= homography(P,Q,R,S, b)
imgfill = np.zeros((img1.shape[0],img1.shape[1],3))
cv2.fillPoly(imgfill,[pts],(255,255,255))
#map pixels in the box to corresponding pixels in fig.d
#d= H^-1*a
result=map_figs(imgfill,img[:,:,:,i], paint, H)
#output
cv2.imwrite('output.jpg',result)

for i in range(1,n_frames,30):
    #norm_pose1, norm_pose2=scale_and_normalize(pose1[i,:,:],pose2[i,:,:])
    mapped_keypoints = map_keypoints(pose2[i,:,:],H)
    score= compare(pose1[i,:,:], mapped_keypoints)
    print('frame ', i, ', distance=',score)
  
#cv2.imshow('pose',img_1)


