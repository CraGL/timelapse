// stroke_timemap_collection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

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
#include <math.h>
#include <stdio.h>



#include <zlib.h>


#include "autolink.h"
#include "config.h"
#include "features.h"
#include "forwards.h"
#include "json.h"
#include "reader.h"
#include "value.h"
#include "writer.h"



using namespace std;
using namespace cv;

std::string GetNextNumber( int lastNum )
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(4) << lastNum;
    return ss.str();
}

double Round(double x)
{
	double y;
	//if (x >= floor(x)+0.5) 
	//	y = ceil(x);
 //   else 
	//	y = floor(x);
	y=floor(x+0.5);

	return y;
}

void Uchar2Double(cv::Mat &img, cv::Mat &dst)
{
	int width=img.cols;
	int height=img.rows;
	for(int i = 0;i < height;i++)
	{
		    Vec3b * Mi = img.ptr<Vec3b>(i);
		    Vec3d * Ni = dst.ptr<Vec3d>(i);
			for(int j =0;j < width;j++)
			{
				for(int k=0;k<3;k++)
				{
					Ni[j][k]=(double)(Mi[j][k]/255.0);
				}
			}
	}

}

double difference_Vec3b(Vec3b A, Vec3b B)
{
	return(sqrt(double((A[0]-B[0])*(A[0]-B[0])+(A[1]-B[1])*(A[1]-B[1])+(A[2]-B[2])*(A[2]-B[2]))));
}


void PD_alpha_interpolation(Mat firstframe,Mat eachframe,Mat &T_frame,double threshold, int currentT,Mat &Paint,Mat &Alpha)
{
	int width=firstframe.cols;
	int height=firstframe.rows;
    Mat _firstframe=Mat::zeros(height,width,CV_64FC3);
	Mat _eachframe=Mat::zeros(height,width,CV_64FC3);
	Mat firstframe_lab;
	Mat eachframe_lab;
	cvtColor(firstframe,firstframe_lab,CV_BGR2Lab);
	cvtColor(eachframe,eachframe_lab,CV_BGR2Lab);

	Uchar2Double(firstframe,_firstframe);
	Uchar2Double(eachframe,_eachframe);

	for(int i=0;i<height;i++)
	{
		Vec3d * Bi=_firstframe.ptr<Vec3d>(i);
		Vec3d * Ai=_eachframe.ptr<Vec3d>(i);

		Vec3b *_Bi=firstframe_lab.ptr<Vec3b>(i);
		Vec3b *_Ai=eachframe_lab.ptr<Vec3b>(i);

		Vec3b * Pi=Paint.ptr<Vec3b>(i);
		Vec3b * Ui=Alpha.ptr<Vec3b>(i);
 
		double alpha=0.0;

		for(int j=0;j<width;j++)
		{
			double d=difference_Vec3b(_Ai[j],_Bi[j]);
			if(d>threshold)
			{
				vector<double> alpha_list;
				for(int k=0;k<3;k++)
				{
					if(Bi[j][k]!=0)
					{
						double value=(Bi[j][k]-Ai[j][k])/Bi[j][k];
						if((value<=1.0)&&(value>=0.0))
							alpha_list.push_back(value);
					}
					if((1.0-Bi[j][k])!=0)
					{
						double value=(Ai[j][k]-Bi[j][k])/(1.0-Bi[j][k]);
						if((value<=1.0)&&(value>=0.0))
							alpha_list.push_back(value);
					}
					//if(Bi[j][k]!=0)
					//	if(alpha<=(Bi[j][k]-Ai[j][k])/Bi[j][k])
					//		alpha=(Bi[j][k]-Ai[j][k])/Bi[j][k];
					//if((1.0-Bi[j][k])!=0)
					//	if(alpha<=(Ai[j][k]-Bi[j][k])/(1.0-Bi[j][k]))
					//		alpha=(Ai[j][k]-Bi[j][k])/(1.0-Bi[j][k]);
				}

				//cout<<alpha_list.size()<<endl;

				if(alpha_list.size()==0)
				{
					cout<<"error! "<<endl;
					getchar();
				}

				std::sort(alpha_list.begin(),alpha_list.end());

				alpha=alpha_list[alpha_list.size()-1];// find max value
 
				 

				for(int k=0;k<3;k++)
				{
					Pi[j][k]=Round(((Ai[j][k]-Bi[j][k])/alpha+Bi[j][k])*255);
					Ui[j][k]=Round(alpha*255);
				}
 
				T_frame.ptr<double>(i)[j]=T_frame.ptr<double>(i)[j]*(1-alpha) + alpha*currentT;
			}
	 
		}
	}

}


Mat toLab( const cv::Mat& bgr )
{
	using namespace cv;
	
	Mat bgr32, lab;
	bgr.convertTo(bgr32,CV_32FC3,1.0/255.0);
	return bgr32;
	//cvtColor(bgr32,lab,CV_BGR2Lab);
	//return lab;
}

Mat diff( cv::Mat mat1, cv::Mat mat2 )
{
    using namespace cv;
    
	Mat diff;
	Mat mat1_lab=toLab(mat1);
	Mat mat2_lab=toLab(mat2);

	subtract( mat1_lab, mat2_lab, diff );

	std::vector<Mat> lab;
	split(diff,lab);

	Mat L2=lab[0].mul(lab[0]);
	Mat a2=lab[1].mul(lab[1]);
	Mat b2=lab[2].mul(lab[2]);

	Mat DIFF = Mat::zeros(diff.rows, diff.cols, CV_32FC1);


	for(int i=0; i<diff.rows;i++)
		for(int j=0; j<diff.cols; j++)
			DIFF.ptr<float>(i)[j]=sqrt( L2.ptr<float>(i)[j]+ a2.ptr<float>(i)[j]+ b2.ptr<float>(i)[j])/1.8;//376.0;/*/1.8;*////376.0;///1.8;///376.0;


	Mat imgout;
	DIFF.convertTo(imgout,CV_8UC1,255.0);
	return imgout;
}

 
double KM_equation(double R, double T, double b)
{
	//cout<<"b"<<b<<endl;
	return pow((R+T*T*b/(1-R*b)-b),2);
}
 
void KM_computation(Mat before,Mat after,double threshold,Mat &R, Mat &T, Mat &timemap, int CurrentTime)
{
	int width=before.cols;
	int height=before.rows;
    Mat _before=Mat::zeros(height,width,CV_64FC3);
	Mat _after=Mat::zeros(height,width,CV_64FC3);
	Mat before_lab;
	Mat after_lab;
	cvtColor(before,before_lab,CV_BGR2Lab);
	cvtColor(after,after_lab,CV_BGR2Lab);

	Uchar2Double(before,_before);
	Uchar2Double(after,_after);

	for(int i=0;i<height;i++)
	{
		Vec3d * Bi=_before.ptr<Vec3d>(i);
		Vec3d * Ai=_after.ptr<Vec3d>(i);

		Vec3b * BBi=before.ptr<Vec3b>(i);
		Vec3b * AAi=after.ptr<Vec3b>(i);

		Vec3b *_Bi=before_lab.ptr<Vec3b>(i);
		Vec3b *_Ai=after_lab.ptr<Vec3b>(i);

		Vec3d * Pi=R.ptr<Vec3d>(i); // R and T are double now!
		Vec3d * Ui=T.ptr<Vec3d>(i);

		for(int j=0;j<width;j++)
		{

			for(int k=0;k<3;k++)
			{
				double R,T;

				if(Bi[j][k]==0.0)
				{

					R=Ai[j][k];
					T=1.0-R;
					//if(CurrentTime==4782)
					//{
					//	cout<<"situation 1: "<<i<<" "<<j<<"  "<<k<<"  "<<CurrentTime<<"  "<<R<<"  "<<T<<endl;
					//	getchar();
					//}
				}
				else if(Ai[j][k]==0.0)
				{
					R=0.0;
					T=0.0;
				}
				else
				{
					if((Ai[j][k]+1.0/Bi[j][k])<=2)
					{
						R=0.0;
						T=sqrt(max(Ai[j][k]/Bi[j][k],0.0));
					}
					else
					{
						if(Ai[j][k]<=Bi[j][k])
						{
							R=0.0;
							T=sqrt(max(Ai[j][k]/Bi[j][k],0.0));
						}
						else
						{

							R=(Ai[j][k]/Bi[j][k]-1)/(Ai[j][k]+1.0/Bi[j][k]-2.0);
							T=sqrt(max((R-Ai[j][k])*(R-1/Bi[j][k]),0.0));

							if(R>(1.0+1/512.0))
							{
								cout.precision(15);
								cout<<"error1: "<<fixed<<R<<endl;
								getchar();
							}
							else if((R>1.0)&&(R<(1.0+1.0/512.0)))
								R=1.0;

							if((R+T)>(1.0+1.0/512))
							{
								cout.precision(15);
								cout<<"error2: "<<fixed<<R+T<<endl;
								getchar();
							
							}
							else if(((R+T)>1.0)&&((R+T)<(1.0+1.0/512)))
								T=1.0-R;

						}
					}
					
				}

				Pi[j][k]=R;
				Ui[j][k]=T;
			 
			}
 
		}
	}

}

void writeMatToFile(const Mat &I, string path) 
{
    //load the matrix size
    int matWidth = I.size().width, matHeight = I.size().height;
 
    //read type from Mat
    int type = I.type();
    gzFile file = gzopen(path.c_str(),"wb");
 
    //write type and size of the matrix first
    gzwrite(file,(const char*) &type, sizeof(type));
    gzwrite(file,(const char*) &matWidth, sizeof(matWidth));
    gzwrite(file,(const char*) &matHeight, sizeof(matHeight));
 
    //write data depending on the image's type
    switch (type)
    {
    default:
        cout << "Error: wrong Mat type: must be CV_32F, CV_64F, CV_32FC3 or CV_64FC3" << endl;
        break;
 
    case CV_64FC3:
       // cout << "Writing CV_64FC3 image" << endl;
       gzwrite(file,(const char*) &I.at<Vec3d>(0)[0],sizeof(double)*3*matWidth*matHeight);
        break;
 
    }
 
    gzclose(file);
}
 
void readFileToMat(Mat &I, string path) 
{
 
	 
    //declare image parameters
    int matWidth, matHeight, type;

	Vec3d vdvalue;
 
    gzFile file =gzopen(path.c_str(),"rb");
 
 

    //read type and size of the matrix first
    gzread(file,(char*) &type, sizeof(type));
    gzread(file,(char*) &matWidth, sizeof(matWidth));
    gzread(file,(char*) &matHeight, sizeof(matHeight));
 
    //change Mat type
    I = Mat::zeros(matHeight, matWidth, type);
 
    //write data depending on the image's type
    switch (type)
    {
    default:
        cout << "Error: wrong Mat type: must be CV_32F, CV_64F, CV_32FC3 or CV_64FC3" << endl;
        break;
 
    case CV_64FC3:
        //cout << "Reading CV_64FC3 image" << endl;
        for (int i=0; i < matWidth*matHeight; ++i) 
		{
            gzread(file,(char*) &vdvalue, sizeof(vdvalue));
            I.at<Vec3d>(i) = vdvalue;
        }
        break;
 
    }
 
	gzclose(file);
 
}


void process_KM_stroke_extraction(const Json::Value &params)
{
	string source_name=params["source_name"].asString();
	VideoCapture capture=VideoCapture(source_name);
	double threshold=params["threshold"].asDouble();
	int frame_num=params["frame_num"].asInt();
	string final_replay_output_path=params["final_replay_output_path"].asString();
	string KM_R_video_name=params["KM_R_video_name"].asString();
	string KM_T_video_name=params["KM_T_video_name"].asString();
	string KM_stroke_video_name=params["KM_stroke_video_name"].asString();
	string firstframe_path=params["firstframe_path"].asString();
	string KM_stroke_image_base_path=params["KM_stroke_image_base_path"].asString();


	int width=params["width"].asInt();
	int height=params["height"].asInt();
	double FPS=params["FPS"].asDouble();
	string R_gzfile_base_name=params["R_gzfile_base_name"].asString();
    string T_gzfile_base_name=params["T_gzfile_base_name"].asString();

	Mat frame;
	capture.read(frame);
	Mat first=frame.clone();
	Mat timemap=Mat::zeros(frame.rows,frame.cols,CV_64FC1);

	 
	VideoWriter writer_R=VideoWriter(KM_R_video_name,CV_FOURCC('D','I','V','X'),FPS,Size(width,height),1);
	VideoWriter writer_T=VideoWriter(KM_T_video_name,CV_FOURCC('D','I','V','X'),FPS,Size(width,height),1);
	
	for(int i=1;i<frame_num;i++)
	{
		//cout<<i<<endl;
		Mat _R=Mat::zeros(frame.rows,frame.cols,CV_64FC3);
	    Mat _T=Mat::zeros(frame.rows,frame.cols,CV_64FC3);
	   
		Mat R=Mat::zeros(frame.rows,frame.cols,CV_8UC3);
	    Mat T=Mat::zeros(frame.rows,frame.cols,CV_8UC3);

		capture.read(frame);
		KM_computation(first,frame,threshold,_R,_T,timemap,i);// _R, _T double!

		_R.convertTo(R,CV_8UC3,255.0);
		_T.convertTo(T,CV_8UC3,255.0);

		//imwrite(R_gzfile_base_name+GetNextNumber(i)+".png",R);
		//imwrite(T_gzfile_base_name+GetNextNumber(i)+".png",T);
		writer_R<<R;
		writer_T<<T;

		writeMatToFile(_R,R_gzfile_base_name+GetNextNumber(i)+".gz");
		writeMatToFile(_T,T_gzfile_base_name+GetNextNumber(i)+".gz");

		first=frame.clone();
	}

 
	VideoWriter writer_stroke=VideoWriter(KM_stroke_video_name,CV_FOURCC('D','I','V','X'),FPS,Size(width,height),1);

	Mat frame2;
	Mat frame3;
 
	Mat firstframe=imread(firstframe_path);
	
	Mat replay=firstframe.clone();

	Mat replay_double=Mat::zeros(replay.size(),CV_64FC3);
	replay.convertTo(replay_double,CV_64FC3,1.0/255.0);

	Mat stroke=Mat::zeros(replay.size(),CV_8UC3);
 
	for(int i=1;i<frame_num;i++)
	{
	    readFileToMat(frame2,R_gzfile_base_name+GetNextNumber(i)+".gz");
		readFileToMat(frame3,T_gzfile_base_name+GetNextNumber(i)+".gz");

		for(int m=0;m<frame2.rows;m++)
		{
			Vec3d *Fm=frame2.ptr<Vec3d>(m); //R double 
			Vec3d *Tm=frame3.ptr<Vec3d>(m); //T double
			Vec3d *Mm=replay_double.ptr<Vec3d>(m);// double!

			Vec3b *Qm=replay.ptr<Vec3b>(m);
			Vec3b *Sm=stroke.ptr<Vec3b>(m);
 
			for(int n=0;n<frame2.cols;n++)
			{
				for(int k=0;k<3;k++)
				{
					//replay
					double value=Fm[n][k]+Tm[n][k]*Tm[n][k]*Mm[n][k]/(1.0-Fm[n][k]*Mm[n][k]);
					if(value>1.0)
					{
						cout.precision(15);
						cout<<"value:    "<<fixed<<value<<endl;
					}
					Mm[n][k]=value>1.0? 1.0:value;
					Qm[n][k]=Round(Mm[n][k]*255);

					//stroke. 
					double value2=Fm[n][k]+Tm[n][k]*Tm[n][k]*1.0/(1.0-Fm[n][k]*1.0);
				 //   if(value2>1.0)
					//{
					//	cout.precision(15);
					//	cout<<"value2:    "<<fixed<<value2<<endl;
					//}
					   
					value2=value2>1.0? 1.0:value2;
					Sm[n][k]=Round(value2*255.0);
				}
			}
		}

        writer_stroke<<stroke;
		imwrite(KM_stroke_image_base_path+GetNextNumber(i)+".png",stroke);
	}

	imwrite(final_replay_output_path,replay);
	cout<<"KM_stroke_finished"<<endl;

}

void process_PD_stroke_extraction(const Json::Value &params)
{
	string source_name=params["source_name"].asString();
	VideoCapture capture=VideoCapture(source_name);
	double threshold=params["threshold"].asDouble();
	int frame_num=params["frame_num"].asInt();
	string PD_paint_base_name=params["PD_paint_base_name"].asString();
	string PD_alpha_base_name=params["PD_alpha_base_name"].asString();
	string PD_stroke_base_name=params["PD_stroke_base_name"].asString();
	string final_replay_output_path=params["final_replay_output_path"].asString();
	string firstframe_path=params["firstframe_path"].asString();
	string PD_stroke_video_name=params["PD_stroke_video_name"].asString();
	int width=params["width"].asInt();
	int height=params["height"].asInt();
	double FPS=params["FPS"].asDouble();


	Mat frame;
	capture.read(frame);
	Mat first=frame.clone();
 
	Mat T_frame=Mat::zeros(frame.rows,frame.cols,CV_64FC1);
	
	for(int i=1;i<frame_num;i++)
	{
		Mat Paint=Mat::zeros(frame.rows,frame.cols,CV_8UC3);
	    Mat Alpha=Mat::zeros(frame.rows,frame.cols,CV_8UC3);
	   
		capture.read(frame);
		PD_alpha_interpolation(first,frame,T_frame,threshold,i,Paint,Alpha);

		imwrite(PD_paint_base_name+GetNextNumber(i)+".png",Paint);
		imwrite(PD_alpha_base_name+GetNextNumber(i)+".png",Alpha);

		first=frame.clone();
	}

 
	VideoWriter writer_stroke=VideoWriter(PD_stroke_video_name,CV_FOURCC('D','I','V','X'),FPS,Size(width,height),1);
	VideoCapture capture2=VideoCapture(PD_paint_base_name+"%04d.png");
	VideoCapture capture3=VideoCapture(PD_alpha_base_name+"%04d.png");
	Mat frame2;
	Mat frame3;
	
	Mat firstframe=imread(firstframe_path);
	
	Mat replay=firstframe.clone();
	Mat stroke;
	Mat replay_double=Mat::zeros(replay.size(),CV_64FC3);
 

	replay.convertTo(replay_double,CV_64FC3,1.0/255.0);

	for(int i=1;i<frame_num;i++)
	{
		//cout<<i<<endl;
		capture2.read(frame2);
		capture3.read(frame3);
		stroke=Mat::zeros(frame2.rows,frame2.cols,CV_8UC3);

		for(int m=0;m<frame2.rows;m++)
		{
			Vec3b *Fm=frame2.ptr<Vec3b>(m);
			Vec3b *Tm=frame3.ptr<Vec3b>(m);
			Vec3b *Qm=replay.ptr<Vec3b>(m);
			Vec3d *_Qm=replay_double.ptr<Vec3d>(m);
			Vec3b *Pm=stroke.ptr<Vec3b>(m);
			for(int n=0;n<frame2.cols;n++)
			{
				for(int k=0;k<3;k++)
				{
					_Qm[n][k]=_Qm[n][k]*(1-Tm[n][k]/255.0)+(Tm[n][k]/255.0)*(Fm[n][k]/255.0);
					Qm[n][k]=Round(_Qm[n][k]*255.0);
					Pm[n][k]=Round(255*(1-Tm[n][k]/255.0)+(Tm[n][k]/255.0)*Fm[n][k]);
				}
			}
		}
 
		writer_stroke<<stroke;
	    imwrite(PD_stroke_base_name+GetNextNumber(i)+".png",stroke);
	}

	imwrite(final_replay_output_path,replay);
	cout<<"PD_stroke_finished"<<endl;


}

 
std::string get_file_contents(const char *filename)
{
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  if (in)
  {
    std::string contents;
    in.seekg(0, std::ios::end);
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);
    in.read(&contents[0], contents.size());
    in.close();
    return(contents);
  }
  throw(errno);
}


int main(int argc, const char* argv[])
{

	if( argc != 2 )
    {
        std::cerr << "Usage: " << argv[0] << "path/to/params.json\n";
        return -1;
    }

	const char* json_name = argv[1];
	std::cerr<<json_name<<endl;

	std::string str=get_file_contents(json_name);
	cout<<str;

	Json::Value root;
	Json::Reader reader;
	bool parsedSuccess = reader.parse(str, root, true);

	if(! parsedSuccess)
	{
		// Report failures and their locations 
		// in the document.
		cout<<"Failed to parse JSON"<<endl 
			<<reader.getFormatedErrorMessages()
			<<endl;

		getchar();
		return -2;
	}

	const Json::Value operations = root["operations"];
    if( operations.size() == 0 )
    {
        std::cerr << "No operations!\n";
        return -3;
    }

	for(int i=0;i<operations.size();i++)
	{
		if(operations[i]["type"].asString()==std::string("KM_stroke_extraction"))
			process_KM_stroke_extraction(operations[i]["params"]);
		if(operations[i]["type"].asString()==std::string("PD_stroke_extraction"))
			process_PD_stroke_extraction(operations[i]["params"]);

	}


	return 0;
}

