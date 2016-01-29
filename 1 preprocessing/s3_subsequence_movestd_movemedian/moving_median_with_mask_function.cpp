#include "stdafx.h"
#include "new_rolling_median.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/ml/ml.hpp>
 
 
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stack>

using namespace cv;

int main(int argc, const char* argv[]) // argv[1] maskname. argv[2] imgname. argv[3] buffersize. argv[4] outputname.
{
    Mat mask=imread(argv[1],1);//F:\\mask_rose_270.png

    Mat img=imread(argv[2],1);//F:\\img_rose_270,png
 
    mm_handle **mm;
    value_t **Buffer;
    int kBufferSize;

    Mat output=img.clone();

    kBufferSize=atoi(argv[3]); // 9 for rose and egg
   
    mm=new mm_handle *[img.rows*3]; // for each row, and three channels

	for(int i=0;i<img.rows*3;i++)
        mm[i] = mm_new(kBufferSize);

	Buffer=new value_t *[img.rows*3];

	for(int i=0;i<img.rows*3;i++)
		Buffer[i]=new value_t[kBufferSize];


	for(int i=0;i<img.rows;i++)
	{
		Vec3b *Mi=img.ptr<Vec3b>(i);
		for(int j=0;j<kBufferSize;j++)
		{
			Buffer[i*3+0][j]=Mi[0][0];
			Buffer[i*3+1][j]=Mi[0][1];
			Buffer[i*3+2][j]=Mi[0][2];
 
		}
	}

 
    for(int i=0;i<img.rows;i++)
		for(int k=0;k<3;k++)
			for(int q=0;q<kBufferSize; q++)
				mm_insert_init( mm[i*3+k], Buffer[i*3+k][q]);

    unsigned char *temp=new unsigned char[kBufferSize+kBufferSize/2+1];

	for(int i=0;i<img.rows;i++)
	{
		Vec3b *Mi=img.ptr<Vec3b>(i);
		Vec3b *Ni=mask.ptr<Vec3b>(i);
		Vec3b *Qi=output.ptr<Vec3b>(i);
        for(int k=0;k<3;k++)
		{
			for(int index=0;index<kBufferSize+kBufferSize/2+1;index++)
				temp[index]=Mi[0][k];
 
		    for(int j=0;j<img.cols; j++)
			{
			 
				int count=0;

				for(int q=0;q<=kBufferSize/2;q++)
				{
 
					if((j+q)>=img.cols)
					{
						temp[kBufferSize+count]=Mi[img.cols-1][k];
					    count++;
					}
 
					else if(Ni[j+q][k]!=255)
					{
						
						temp[kBufferSize+count]=Mi[j+q][k];
						count++;
					}
					 
				}

                for(int q=1;q<=kBufferSize;q++)
				{
					if((j-q)<0)
					{
					 	temp[kBufferSize-q]=Mi[0][k];
				 
					}
					else if(Ni[j-q][k]!=255)
					{
						temp[kBufferSize-q]=Qi[j-q][k];
						 
					}

				}

				for(int q=count;q<kBufferSize+count;q++)
				{
 
					mm_update(mm[i*3+k],temp[q]);
				}
 

			    Qi[j][k]=mm_get_median(mm[i*3+k]);
			}

 
		}
	}
	imwrite(argv[4],output);//"F:\\rose_270_recover_movingmedian_buffersize_09_new.png"
}
