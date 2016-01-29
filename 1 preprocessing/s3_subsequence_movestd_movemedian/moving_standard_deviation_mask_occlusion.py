INPUT_PATH_PREFIX = "rose_"
OUTPUT_PATH_PREFIX = "rose_"
#### These indices are the keyframe indices of the input sequence.
KEYFRAME_INDICES=np.array([0,255,291,298,361,591,692,779,835,1024,1130,
                              1142,1509,1753,1790,2037,2097,2101,2456,3116,
                              3492,3576,3754,3981,4029,4051,4199,4219,4267,
                              4574,4627,4685,4813,4894,4917,4968,4974])

import cv2
import numpy as np
import bottleneck
import subprocess

#function that modifies those functions in bottleneck module
def bottleneck_centered( func, data, window, axis = -1 ):
    assert window % 2 == 1
    result = func( data, window = window, axis = axis )
    shift = window//2
    result = np.roll( result, -shift, axis = axis )
    ## I don't know how to write a general axis selection, so let's use swapaxis
    if -1 != axis: result = result.swapaxes( axis, -1 )
    result[ ...,:shift ] = result[ ..., shift ][...,np.newaxis]
    result[ ..., -shift-1: ] = result[ ..., -shift-1 ][...,np.newaxis]
    if -1 != axis: result = result.swapaxes( axis, -1 )
    return result
    
def lerp_image(first,last,index,length,mask): #mask !=0 means common part.
    interpolation=np.zeros(first.shape,dtype=np.uint8)
    if index>length:
        index=length
    interpolation=np.uint8(first*(1.0-index*1.0/length)+last*(index*1.0/length)).clip(0,255)
    return interpolation
    
print KEYFRAME_INDICES.shape


capture=cv2.VideoCapture( INPUT_PATH_PREFIX + "subsequence_colorshift_%04d.png")
capture1=cv2.VideoCapture( INPUT_PATH_PREFIX + "keyframe_mask_%04d.png")
capture2=cv2.VideoCapture( INPUT_PATH_PREFIX + "subsequence_last_%04d.png")

ret,firstframe=capture.read()

first=firstframe
firstframe2=firstframe

first_keyframe=firstframe
ret2,last_keyframe=capture2.read()

kbuffersize=9
count=0
std_threshold=1.5
window = 7
window_mov_mean=3

kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((3,3),np.uint8)

recover_image_output_path=OUTPUT_PATH_PREFIX + "subsequence_movestd_movingmedian_recover"

for index in range(1,KEYFRAME_INDICES.shape[0]):
    
    retval1,mask=capture1.read()
    
    mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    
    cv2.imwrite(OUTPUT_PATH_PREFIX + "keyframe_mask_closed_"+'{0:04}'.format(index)+".png",mask)
 
    nonzero_num=cv2.countNonZero(mask)
    print nonzero_num
    print mask.shape
    
    if nonzero_num<mask.shape[0]*mask.shape[1]:
 
        zero_num=firstframe2.shape[0]*firstframe2.shape[1]-nonzero_num
        img=np.zeros((zero_num,KEYFRAME_INDICES[index]-KEYFRAME_INDICES[index-1]+kbuffersize+1,3),dtype=np.uint8)
        temp=np.zeros((firstframe2.shape[0],firstframe2.shape[1],3),dtype=np.uint8)

        for i in range(0,KEYFRAME_INDICES[index]-KEYFRAME_INDICES[index-1]+1):
            if i==0:
                frame=firstframe
            if i>=1:
                retval,frame=capture.read()
            frame_lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
            img[:,i,:]=(frame_lab[mask==0]).reshape((zero_num,3))
            temp=frame_lab

        for i in range(KEYFRAME_INDICES[index]-KEYFRAME_INDICES[index-1]+1,KEYFRAME_INDICES[index]-KEYFRAME_INDICES[index-1]+kbuffersize+1):
            img[:,i,:]=(temp[mask==0]).reshape((zero_num,3))

        firstframe=cv2.cvtColor(temp,cv2.COLOR_LAB2BGR)

        img_RGB= cv2.cvtColor(img,cv2.COLOR_LAB2BGR)

        rose_subsequence_img_name=OUTPUT_PATH_PREFIX + "subsequence_img_"+'{0:04}'.format(index)+".png"
        cv2.imwrite(rose_subsequence_img_name,img_RGB)

        stdevs=bottleneck_centered(bottleneck.move_std,img,window=window,axis=1)
        
        stdevs_per_pixel = stdevs.sum( axis = 2 )/(stdevs.shape[2])

        highlighted_bad_pixels = img_RGB.copy()
        badmask=np.zeros((zero_num,KEYFRAME_INDICES[index]-KEYFRAME_INDICES[index-1]+kbuffersize+1),dtype=np.uint8)

        badmask[ stdevs_per_pixel > std_threshold] = 255

        rose_subsequence_mask_name=OUTPUT_PATH_PREFIX + "subsequence_mask_"+'{0:04}'.format(index)+".png"
        cv2.imwrite(rose_subsequence_mask_name,badmask)

        highlighted_bad_pixels[ stdevs_per_pixel > std_threshold ] = (255,0,0)
        rose_subsequence_highlighted_name=OUTPUT_PATH_PREFIX + "subsequence_highlighted_"+'{0:04}'.format(index)+".png"
        cv2.imwrite(rose_subsequence_highlighted_name,highlighted_bad_pixels)


        rose_subsequence_mm_outputname=OUTPUT_PATH_PREFIX + "subsequence_movingmedian_output_"+'{0:04}'.format(index)+".png"

        
        #recover by moving masked_median using C code
        subprocess.call(['moving_median_with_mask_function.exe', rose_subsequence_mask_name, rose_subsequence_img_name,str(kbuffersize),rose_subsequence_mm_outputname])

        frame_mask=mask
        recover_base=cv2.imread(rose_subsequence_mm_outputname)
        
        
         #using bilateral filtering
        temp=np.zeros((recover_base.shape[0]*3,recover_base.shape[1],recover_base.shape[2]),dtype=np.uint8)
        for i in range(0,recover_base.shape[0]):
            temp[3*i+0,:,:]=recover_base[i,:,:]
            temp[3*i+1,:,:]=recover_base[i,:,:]
            temp[3*i+2,:,:]=recover_base[i,:,:]
            
        temp2=cv2.adaptiveBilateralFilter(temp,(3,15),200.0)
        
        for i in range(0,recover_base.shape[0]):
            recover_base[i,:,:]=temp2[3*i+1,:,:]

        _frame=np.zeros(firstframe.shape,dtype=np.uint8)

        for i in range (0,KEYFRAME_INDICES[index]-KEYFRAME_INDICES[index-1]+kbuffersize+1):
           
            _frame=lerp_image(first_keyframe,last_keyframe,i,KEYFRAME_INDICES[index]-KEYFRAME_INDICES[index-1],frame_mask)
            _frame[frame_mask==0]=(recover_base[:,i,:]).reshape((zero_num,3))
            cv2.imwrite(recover_image_output_path+'{0:04}'.format(count)+".png",_frame)
            count=count+1

        firstframe2=firstframe

        print count   
        
    else:
        
        _frame=np.zeros(firstframe.shape,dtype=np.uint8)
        frame_mask=mask
        for i in range(0,KEYFRAME_INDICES[index]-KEYFRAME_INDICES[index-1]+1):
            if i==0:
                frame=firstframe
            if i>=1:
                retval,frame=capture.read()
            firstframe=frame
        
        for i in range(0,KEYFRAME_INDICES[index]-KEYFRAME_INDICES[index-1]+1):
            
            _frame=lerp_image(first_keyframe,last_keyframe,i,KEYFRAME_INDICES[index]-KEYFRAME_INDICES[index-1],frame_mask)
            cv2.imwrite(recover_image_output_path+'{0:04}'.format(count)+".png",_frame)
            count=count+1
            
        firstframe2=firstframe
        print count 
    
    first_keyframe=last_keyframe
    ret2,last_keyframe=capture2.read()

 
