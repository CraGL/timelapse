//#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/photo/photo.hpp>

 
#include <Eigen/Sparse> 
#include <Eigen/SparseCholesky>
#include <Eigen/LU>
#include <unsupported/Eigen/IterativeSolvers>


//json library
#include "autolink.h"
#include "config.h"
#include "features.h"
#include "forwards.h"
#include "json.h"
#include "reader.h"
#include "value.h"
#include "writer.h"



#include <list>
#include <vector>
#include <map>
#include <stack>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <array>
#include <string>
#include <streambuf>
#include <iomanip>
#include <sstream>


using namespace std;
using namespace cv;

namespace
{
    const bool SHOW_IMAGE = false;
}

/// Superclass
class VideoFilter
{
public:
    VideoFilter() : m_output(0) {}
    virtual ~VideoFilter() {}
    
	void SetOutput( VideoFilter* out ) { m_output = out; }
    
    Mat NextFrame( const Mat& frame )
    {
        Mat result=Mat::zeros(frame.rows,frame.cols,CV_8UC3);

        bool success = this->doNextFrame( frame, result );
        if( !success ) return result;

		
        
        assert( !result.empty() );
        
        if( m_output )
        {
			return m_output->NextFrame( result );
		}
		else
		{
			return result;
		}
    }
    
protected:
    // subclasses fill this in.
    // Return false to skip this frame, true otherwise.
    virtual bool doNextFrame( const Mat& inputFrame, Mat& outputFrame ) = 0;
    
private:
    VideoFilter* m_output;
};


class VideoReader : public VideoFilter
{
public:
    VideoReader( const Json::Value& params )
    {
        m_video = VideoCapture( params["filename"].asString() );
		count=0;
		FrameStop=params["frameStop"].asInt();
		FrameStart=params["frameStart"].asInt();
		outputFrame=Mat::zeros((int)m_video.get(CV_CAP_PROP_FRAME_HEIGHT),(int)m_video.get(CV_CAP_PROP_FRAME_WIDTH),CV_8UC3);
    }
    
    bool IsFinished() const
    {
        return count>FrameStop? true: false;
    }

	int getcount()
	{
		return count;
	}
    
protected:
    bool doNextFrame( const Mat&, Mat& outputFrame ) override
    {
		if(count<FrameStart)
	     {
			 while(count<FrameStart)
			 {
				 m_video.read(outputFrame);
				 count++;
			 }
			 //outputFrame = outputFrame(cv::Rect(0,0,3,3));
			 return true;
	     }
		else
		{		 m_video.read(outputFrame);
				 count++;
				 //outputFrame = outputFrame(cv::Rect(0,0,3,3));
				 return true;
		}
    }

private:
     VideoCapture m_video;
	 Mat outputFrame;
	 int count;
	 int FrameStop;
	 int FrameStart;
};


class ColorShift: public VideoFilter
{
public:
	ColorShift(const Json::Value& params)
	{
		percent=params["Percent"].asDouble();
		Threshold=params["Threshold"].asDouble();
	
	}
protected:
	
	virtual bool doNextFrame( const Mat& inputFrame, Mat& outputFrame) override
	{
		if( fixedFrame.empty() )
		{
			fixedFrame = inputFrame.clone();
			outputFrame = fixedFrame.clone();
			return true;
		}

		Mat current_diff = Mat::zeros(inputFrame.rows,inputFrame.cols, CV_8UC1);
		
		current_diff = diff(fixedFrame,inputFrame);

 		int height=inputFrame.rows;
		int width=inputFrame.cols;

		//count the non zero element, and do not change tframe corresponding to the frame whose number >percent% 
		Mat temp;
		threshold(current_diff,temp,Threshold,1.0,THRESH_BINARY);
		if(countNonZero(temp)>percent*height*width)
			 ColorShiftRecover(inputFrame,Threshold,outputFrame,fixedFrame);
		else
			outputFrame=inputFrame.clone();

		return true;
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
				DIFF.ptr<float>(i)[j]=sqrt( L2.ptr<float>(i)[j]+ a2.ptr<float>(i)[j]+ b2.ptr<float>(i)[j])/1.8;//376.0;//1.8;//376.0;/*/1.8;*////376.0;///1.8;///376.0;


		Mat imgout;
		DIFF.convertTo(imgout,CV_8UC1,255.0);
		//imshow("imgout",imgout);
		//waitKey(0);
		return imgout;

	}



	void ColorShiftRecover(const Mat& inputFrame, double Threshold, Mat& outputFrame, Mat& fixedFrame)
	{
		int height=inputFrame.rows;
		int width=inputFrame.cols;
	    std::vector<Mat> RGB_old;
	    split(fixedFrame,RGB_old); //fixedframe is last old frame
		Mat R_old=RGB_old[0];
		Mat G_old=RGB_old[1];
		Mat B_old=RGB_old[2];

	    std::vector<Mat> RGB_new;
	    split(inputFrame,RGB_new); //inputframe is new frame.
		Mat R_new=RGB_new[0];
		Mat G_new=RGB_new[1];
		Mat B_new=RGB_new[2];

		Mat R_recover,G_recover,B_recover,RGB_recover;

		
		std::vector<int> INDEX;

		Mat mask;


#define MASK_CHOICE 0   // 0 for sorting diff color value and then choose a range. 1 for sorting old and new imgs' color. and then compute diff. then sort diff and then choose a range.
#if MASK_CHOICE==0
 
		mask=Mat::zeros(height,width,CV_8UC1);
		getmask(fixedFrame,inputFrame,mask);  // get the mask of the selected pixels that are in  N/8 to N/4 range of color differences .

#elif MASK_CHOICE==1

		 
		std::vector<double> Diff_R,Diff_G,Diff_B;

		getsortedDiff(R_old,R_new,Diff_R);
		getsortedDiff(G_old,G_new,Diff_G);
 		getsortedDiff(B_old,B_new,Diff_B);

		std::vector<std::pair<double,int>> DIFF;
		for(int i = 0;i <Diff_R.size();i++)
	    {
			DIFF.push_back(std::make_pair(std::sqrt(Diff_R[i]*Diff_R[i]+Diff_G[i]*Diff_G[i]+Diff_B[i]*Diff_B[i]),i));
		}


struct sort_pred {
    bool operator()(const std::pair<double,int> &left, const std::pair<double,int> &right) {
        return left.first < right.first;
    }
    };

		sort(DIFF.begin(),DIFF.end(),sort_pred());

		int N=DIFF.size();

		for(int i=N/10; i<N*8/10; i++)
			INDEX.push_back(DIFF[i].second);
		
#endif

		SolveLSM(R_old,R_new,mask,R_recover,INDEX);
		SolveLSM(G_old,G_new,mask,G_recover,INDEX);
		SolveLSM(B_old,B_new,mask,B_recover,INDEX);


		std::vector<Mat> Recover;
		Recover.push_back(R_recover);
		Recover.push_back(G_recover);
		Recover.push_back(B_recover);
		merge(Recover,RGB_recover);

		outputFrame=RGB_recover.clone();
		fixedFrame=outputFrame.clone();
		                                 // since 10/16/2014, we do colorshift based on fixedframe for a small
		                                 // sequence of frame, and do not update the fixedframe. that means, we compare each
		                                 // frame with first frame to do colorshift recovering.

	}


	struct Triple
    {
 
		double diff;
		int row;
		int col;

		Triple(double k, int i, int j) : diff(k), row(i), col(j) {}

		bool operator < (const Triple& str) const
        {
			return (diff < str.diff);
        }
    };


void getmask(Mat _before,Mat _after, Mat &mask)
	{

		Mat before,after;
		_before.convertTo(before,CV_64FC3,1.0/255.0);
		_after.convertTo(after,CV_64FC3,1.0/255.0);

		int height=before.rows;
		int width=before.cols;

		Mat diff=Mat::zeros(height,width,CV_64FC1);

		for(int i=0;i<height; i++)
			for(int j=0;j<width;j++)
				for(int k=0;k<3;k++)
					diff.ptr<double>(i)[j]+=pow(abs(before.ptr<Vec3d>(i)[j][k]-after.ptr<Vec3d>(i)[j][k]),2.0);

 
		std::vector<Triple> diff_list;

 
		
		for(int i=0;i<height; i++)
			for(int j=0;j<width;j++)
			{
				diff_list.push_back(Triple(diff.ptr<double>(i)[j],i,j));
			}


		std::sort(diff_list.begin(),diff_list.end());

		//for(int index=0;index<1000;index++)
		//cout<<(diff_list[index].diff)<<endl;
 

		int N=height*width;
		for(int index=N/8;index<N/4;index++)
			mask.ptr<unsigned char>(diff_list[index].row)[diff_list[index].col]=1;

	}


void getsortedDiff(Mat _OldImg, Mat _NewImg,vector<double> &diff)
	{
		
		Mat OldImg,NewImg;
		_OldImg.convertTo(OldImg,CV_64FC1,1.0/255.0);
		_NewImg.convertTo(NewImg,CV_64FC1,1.0/255.0);

		int height=NewImg.rows;
		int width=NewImg.cols;

		std::vector<double> OLD,NEW;

		for(int i = 0;i <height;i++)
	    {
		    double *Mi = OldImg.ptr<double>(i);
			double *Ni=  NewImg.ptr<double>(i);
 
		    for(int j =0;j <width;j++)
			{
				OLD.push_back(Mi[j]);
				NEW.push_back(Ni[j]);
			}
		}

		sort(OLD.begin(),OLD.end());
		sort(NEW.begin(),NEW.end());
		for(int i=0;i<OLD.size();i++)
			diff.push_back(OLD[i]-NEW[i]);
 
     }


void SolveLSM(Mat _OldImg, Mat _NewImg, Mat &mask, Mat &recover, vector<int> &INDEX)
	{
		Mat OldImg,NewImg;
		_OldImg.convertTo(OldImg,CV_64FC1,1.0/255.0);
		_NewImg.convertTo(NewImg,CV_64FC1,1.0/255.0);

		Eigen::SparseMatrix<double,Eigen::RowMajor> M(2,2);
	    Eigen::VectorXd N(2);
		Eigen::VectorXd X(2);

		int height=OldImg.rows;
		int width= OldImg.cols;

		double count=0;
		double new_sum=0;
		double new_sum_square=0;
		double old_sum=0;
		double old_new_sum=0;

 
#define MASK 1 // 0 for not using mask, 1 for using mask. 2 for using vector INDEX

#if MASK==0
		for(int i = 0;i <height;i++)
	    {
		    double *Mi = OldImg.ptr<double>(i);
			double *Ni=  NewImg.ptr<double>(i);
 
		    for(int j =0;j <width;j++)
			{
				if(abs(Mi[j]-Ni[j])>=0)
				{
					count+=1.0;       
					new_sum+=Ni[j];
					old_sum+=Mi[j];
					new_sum_square+=Ni[j]*Ni[j];
					old_new_sum+=Ni[j]*Mi[j];
				}
			}
		}

#elif MASK==1
		for(int i = 0;i <height;i++)
	    {
		    double *Mi = OldImg.ptr<double>(i);
			double *Ni=  NewImg.ptr<double>(i);
			unsigned char *Maski=mask.ptr<unsigned char>(i);
 
		    for(int j =0;j <width;j++)
			{
				if(Maski[j]==1)
				{
					count+=1.0;       
					new_sum+=Ni[j];
					old_sum+=Mi[j];
					new_sum_square+=Ni[j]*Ni[j];
					old_new_sum+=Ni[j]*Mi[j];
				}
			}
		}





#elif MASK==2

		std::vector<double> OLD,NEW;

		for(int i = 0;i <height;i++)
	    {
		    double *Mi = OldImg.ptr<double>(i);
			double *Ni=  NewImg.ptr<double>(i);
 
		    for(int j =0;j <width;j++)
			{
				OLD.push_back(Mi[j]);
				NEW.push_back(Ni[j]);
			}
		}

		sort(OLD.begin(),OLD.end());
		sort(NEW.begin(),NEW.end());

		for(int i=0;i<INDEX.size();i++)
		{
			count+=1.0;      
			new_sum+=NEW[INDEX[i]];
			old_sum+=OLD[INDEX[i]];
			new_sum_square+=NEW[INDEX[i]]*NEW[INDEX[i]];
			old_new_sum+=NEW[INDEX[i]]*OLD[INDEX[i]];
		}

#endif

 

		M.insert(0,0)=count;
		M.insert(1,0)=new_sum;
		M.insert(0,1)=new_sum;
		M.insert(1,1)=new_sum_square;
		N(0)=old_sum;
		N(1)=old_new_sum;

		Eigen::SimplicialCholesky< Eigen::SparseMatrix<double> > chol(M); //solve MX=N
	    X=chol.solve(N);

		Mat temp_recover=OldImg.clone();

		for(int i = 0;i <height;i++)
	    {
		    double *Mi = OldImg.ptr<double>(i);
			double *Ni=  NewImg.ptr<double>(i);
			double *Qi = temp_recover.ptr<double>(i);
 
		    for(int j =0;j <width;j++)
			{
				if(abs(Mi[j]-Ni[j])>=0)
				{
					double value=(X[0]+X[1]*Ni[j]);  // ||a+b*new-old|| so, recover=a+b*new
					
					if(value>1.0)
						Qi[j]=1.0;
 
					else if(value<0.0)
						Qi[j]=0.0;
					else
						Qi[j]=value; 
				}
			}
		}
		
		temp_recover.convertTo(recover,CV_8UC1,255.0);
	
    }
	

private:

	double percent;
	Mat fixedFrame;
	double Threshold;
};


class VideoWriter : public VideoFilter
{
public:
    VideoWriter( const Json::Value& params )
    {
		m_video = cv::VideoWriter( params["filename"].asString(), CV_FOURCC('D','I','V','X'), params["FPS"].asDouble(), Size(params["width"].asInt(),params["height"].asInt()), true );
    }
	
protected:
    virtual bool doNextFrame( const Mat& inputFrame, Mat& outputFrame )
    {
        outputFrame = inputFrame.clone();
        m_video.write( inputFrame );
        return true;
    }

private:
     cv::VideoWriter m_video;
};


class VideoSaver : public VideoFilter
{
public:
    VideoSaver( const Json::Value& params )
    {
		m_path = params["filename"].asString();
		number=0;
    }
	
protected:
    virtual bool doNextFrame( const Mat& inputFrame, Mat& outputFrame )
    {
		string fullpathname=m_path+ GetNextNumber(number)+".png";
        outputFrame = inputFrame.clone();
        cv::imwrite( fullpathname, inputFrame );
		number++;
		return true;
    }

	std::string GetNextNumber( int lastNum )
    {
      std::stringstream ss;
      ss << std::setfill('0') << std::setw(4) << lastNum;
      return ss.str();
    }

private:
     std::string m_path;
	 int number;
};

VideoFilter* CreateVideoFilterByName( const std::string& filterName, const Json::Value& params )
{
         if( filterName == std::string("VideoReader") ) 	return new VideoReader(params);
    else if( filterName == std::string("VideoWriter") ) 	return new ::VideoWriter(params);
    else if( filterName == std::string("VideoSaver") ) 	    return new VideoSaver(params);
	else if( filterName == std::string("ColorShift") )  	return new ColorShift(params);
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


int main( int argc, const char* argv[] )
{

	if( argc != 2 )
    {
        std::cerr << "Usage: " << argv[0] << "path/to/params.json\n";
        return -1;
    }
	const char *json_name = argv[1];
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


    std::vector< VideoFilter* > filters;
    for( int i = 0; i < operations.size(); ++i )
    {
        VideoFilter* nextFilter = CreateVideoFilterByName( operations[i]["filter"].asString(), operations[i]["params"] );
		filters.push_back( nextFilter );
		if( i > 0 ) filters.at(i-1)->SetOutput( nextFilter );
    }
    
    // The first filter must be a VideoReader
    assert( filters.size() > 0 );
    VideoReader* videoreader = dynamic_cast< VideoReader* >( filters.front() );
    Mat ignore, result;
	int num=0;
    while( !videoreader->IsFinished() )
    {
		cout<<num<<endl;

		result = videoreader->NextFrame( ignore );
		num++;
		    
    }
    
    // Cleanup
    for( int i = 0; i < filters.size(); ++i )
    {
        delete filters.at(i);
    }
    filters.clear();
    
}
