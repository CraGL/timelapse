sequence_name="rose-unrotate-%04d.png"
output_path="rose_albedo_"
num_frames=5334
Reflectance=0.66 #### choose the value that smaller than 1./all_time_maxval.


import numpy as np
import cv2

capture=cv2.VideoCapture(sequence_name)

ret,frame=capture.read()
_first=frame

first_modified=np.zeros(frame.shape,dtype=np.uint8)

first=cv2.medianBlur(_first,55)

first_modified[:,:,0]=first.max(axis=2)
first_modified[:,:,1]=first.max(axis=2)
first_modified[:,:,2]=first.max(axis=2)

all_time_maxval = 0.
 
for i in range(1,num_frames):
    ## Load each frame
    new_frame =frame*1.0/first_modified
    ## Compute the maximum value in any channel after applying per-pixel-scale.
    maxval = new_frame.max()
    ## Find the all-time-maximum value.
    if maxval > all_time_maxval:
        all_time_maxval = maxval
        ## Print it if it changes:
        print all_time_maxval
    ret,frame = capture.read()

print 'Final all time max:', all_time_maxval
print 1./all_time_maxval  ###### for rose, this value is 0.665217391304


capture3=cv2.VideoCapture(sequence_name)
for i in range(0,num_frames):
   
    ret3,frame3=capture3.read()
    
    modified=frame3*255.0*Reflectance/first_modified
    
    test1=modified[:,:,0]
    test2=modified[:,:,1]
    test3=modified[:,:,2]

    if test1[test1>255.0].shape[0]>0:
        print test1[test1>255.0]
 
    if test2[test2>255.0].shape[0]>0:
        print test2[test2>255.0]
        
    if test3[test3>255.0].shape[0]>0:
        print test3[test3>255.0]
        
    division3=(modified).clip(0,255)
 
    cv2.imwrite(output_path+'{0:04}'.format(i)+".png",division3)

print 'finished'
