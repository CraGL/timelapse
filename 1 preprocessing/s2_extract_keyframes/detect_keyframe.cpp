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

#include <Eigen/Sparse> 
#include <Eigen/SparseCholesky>
#include <Eigen/LU>
#include <unsupported/Eigen/IterativeSolvers>
#include <Eigen/Dense>

using namespace cv;
using namespace std;

std::string GetNextNumber( int lastNum )
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(4) << lastNum;
    return ss.str();
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
	getmask(fixedFrame,inputFrame,mask);  // get the mask of the selected pixels that are in  N/4 to N/2 range of color differences .

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
	//fixedFrame=outputFrame.clone();// ?? inputframe.clone or outputframe.clone???????
		                                // since 10/16/2014, we do colorshift based on fixedframe for a small
		                                // sequence of frame, and do not update the fixedframe. that means, we compare each
		                                // frame with first frame to do colorshift recovering.

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
 

void subsequence_colorshift(Mat fixedFrame, Mat inputFrame, Mat &outputFrame)
{
	    double Threshold=0.0;
		double percent=0.5;
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
}


void extract_difference_mask_between_keyframe(std::string keyframe_sequence_name, std::string first_keyframe_name, double keyframe_diff_threshold, int keyframe_num, std::string keyframe_mask_output_path)
{
    VideoCapture capture=VideoCapture(keyframe_sequence_name);
    Mat first=imread(first_keyframe_name,1);
	Mat _frame;
	Mat frame;
	Mat first_lab;
	Mat first_gaussian;
	GaussianBlur(first,first_gaussian,Size(3,3),1.0,1.0);
	cvtColor(first_gaussian,first_lab,CV_BGR2Lab);

	for(int i=0;i<keyframe_num;i++)
	{
		capture.read(_frame);
		frame=_frame.clone();
		Mat frame_gaussian;
        GaussianBlur(frame,frame_gaussian,Size(3,3),1.0,1.0);
		Mat frame_lab;
		cvtColor(frame_gaussian,frame_lab,CV_BGR2Lab);

		Mat diff=Mat::zeros(first.rows,first.cols,CV_8UC1);

		for(int m=0;m<frame.rows;m++)
		{
			Vec3b *Fm=first_lab.ptr<Vec3b>(m);
			Vec3b *Tm=frame_lab.ptr<Vec3b>(m);
		 
			unsigned char *Qm=diff.ptr<unsigned char>(m);
			for(int n=0;n<frame.cols;n++)
			{
				double d=0;
				for(int k=0;k<3;k++)
					d+=(Fm[n][k]-Tm[n][k])*(Fm[n][k]-Tm[n][k]);
				d=sqrt(d);
				if(d<=keyframe_diff_threshold)
					Qm[n]=255;
			}
		}
		first_lab=frame_lab.clone();
		imwrite(keyframe_mask_output_path+GetNextNumber(i)+".png",diff);
	}
}



void prepare_intermediate_data( const std::string& input_sequence, const int N, const std::string& intermediate_data_path_prefix )
{
    double threshold=10;
    double threshold2=5;
    int startindex=0;
    int endindex=N-1;
    int max_num=50;
    string source_sequence=input_sequence;
    int sign_array[N];
    sign_array[0]=0;

	VideoCapture capture=VideoCapture(source_sequence);
	Mat frame;

	capture.read(frame);
	Mat img=frame.clone();
	Mat img_lab;
	cvtColor(img,img_lab,CV_BGR2Lab);
	
	
	for(int i=1;i<=endindex;i++)
	{
		Mat result=Mat::zeros(frame.rows,frame.cols,CV_8UC1);
		capture.read(frame);
		Mat frame_lab;
		cvtColor(frame,frame_lab,CV_BGR2Lab);
		Mat temp;
		subtract(frame_lab,img_lab,temp);

		for(int m=0;m<frame.rows;m++)
		{
			unsigned char *Mm=result.ptr<unsigned char>(m);
			Vec3b *Nm=temp.ptr<Vec3b>(m);
			for(int n=0;n<frame.cols;n++)
			{
			    double diff=sqrt(double(Nm[n][0]*Nm[n][0]+Nm[n][1]*Nm[n][1]+Nm[n][2]*Nm[n][2]));
				if(diff>threshold)
					Mm[n]=255;
			}
		}
		img_lab=frame_lab.clone();
	    cout<<i<<endl;
		string save_path=intermediate_data_path_prefix+"Lab_diff_binary_"+GetNextNumber(i-startindex)+".png";
		imwrite(save_path,result);
	}
 

	string input1=intermediate_data_path_prefix+"Lab_diff_binary_%04d.png";
	VideoCapture capture1=VideoCapture(input1);
	Mat _frame;

	for(int i=1;i<=endindex;i++)
	{
		capture1.read(_frame);
		if(countNonZero(_frame)<max_num)
			sign_array[i]=0;
		else
			sign_array[i]=1;
	}


    int selected_sign[N];
	for(int i=0;i<N;i++)
		selected_sign[i]=0;
	selected_sign[0]=2;
	selected_sign[N-1]=1;
 
 
	for(int i=2;i<endindex-2;i++)
	{
		if((sign_array[i-1]==0)&&(sign_array[i]==0)&&(sign_array[i+1]==0)&&(sign_array[i+2]==1))
		{
			selected_sign[i]=1;
			cout<<"first: "<<i<<"  ";
		}
		if((sign_array[i-1]==1)&&(sign_array[i]==0)&&(sign_array[i+1]==0)&&(sign_array[i+2]==0))
		{
			selected_sign[i]=2;
			cout<<"end: "<<i<<endl;
		}
	}
 
	for(int i=0;i<N;i++)
	{
		if(selected_sign[i]!=0)
		cout<<i<<"   ";
	}

 
	std::string input2=input_sequence;
	VideoCapture capture2=VideoCapture(input2);
	Mat frame2;

	int count=0;
	int j;
	int flag;
	Mat Mean;

    vector<int> averaged_index;

	int last_position;
	for(int i=N-1;i>=0;i--)
	{
		if(selected_sign[i]==1)
		{
			last_position=i;
			break;
		}
	}


    for(int i=0;i<N;)
	{
		if(selected_sign[i]==2)
		{
			cout<<"2   ";
			capture2.read(frame2);
			string good_frame_name=intermediate_data_path_prefix+"good_original_frame_"+GetNextNumber(i)+".png";
			imwrite(good_frame_name,frame2);
			j=i;
			flag=2;
			Mean= Mat::zeros(frame2.rows, frame2.cols,CV_32FC3); 
			accumulate(frame2, Mean);
		}
		if(flag==2)
		{
			i++;
			capture2.read(frame2);
			string good_frame_name=intermediate_data_path_prefix+"good_original_frame_"+GetNextNumber(i)+".png";
			imwrite(good_frame_name,frame2);
			accumulate(frame2, Mean);

			if(selected_sign[i]==1)
			{
				cout<<"1   ";
				averaged_index.push_back(1);
				flag=1;
				Mean = Mean /(i-j+1);
				Mean.convertTo(Mean,CV_8UC3);
				string save_name1=intermediate_data_path_prefix+"averaged_"+GetNextNumber(count)+".png";
				imwrite(save_name1,Mean);
				count++;
			}


		}
		if(flag==1)
		{	
			if(i==last_position)
				break;
			i++;
		    if(selected_sign[i]==0)
			{
				averaged_index.push_back(0);
				capture2.read(frame2);
				string save_name2=intermediate_data_path_prefix+"averaged_"+GetNextNumber(count)+".png";
				imwrite(save_name2,frame2);
				count++;
			}
		}
	}

	cout<<averaged_index.size();
	for(int i=0;i<averaged_index.size();i++)
		if(averaged_index[i]!=0)
			cout<<i<<"  ";

            
	string input3=intermediate_data_path_prefix+"averaged_%04d.png";
	VideoCapture capture3=VideoCapture(input3);
	Mat frame3;
	Mat base;
	capture3.read(frame3);
	Mat output=frame3.clone();
	int flag3;
	imwrite(intermediate_data_path_prefix+"subsequence_colorshift_"+GetNextNumber(0)+".png",output);

	for(int i=0;i<averaged_index.size();)
	{
		//cout<<i<<"  ";
		if(averaged_index[i]==1)
		{
			base=output.clone();
			flag3=0;
		}
		if(flag3==0)
		{
			capture3.read(frame3);
			i++;
			if(i<=averaged_index.size()-1)
			{
				subsequence_colorshift(base,frame3,output);
				string save_name3=intermediate_data_path_prefix+"subsequence_colorshift_"+GetNextNumber(i)+".png";
				imwrite(save_name3,output);
				if(i==averaged_index.size()-1)
					break;
			}
 
		}
	}
    

	string input4=intermediate_data_path_prefix+"subsequence_colorshift_%04d.png";
	VideoCapture capture4=VideoCapture(input4);
	Mat frame4;
	capture4.read(frame4);

	int index=0;
	for(int i=1;i<averaged_index.size();i++)
	{
		//cout<<i<<"  ";
		capture4.read(frame4);
		if(averaged_index[i]==1)
		{
			string save_name4=intermediate_data_path_prefix+"subsequence_last_"+GetNextNumber(index)+".png";
			imwrite(save_name4,frame4);
			index++;
		}
	}
    
    cout<<"get keyframes!"<<endl;
}

int main()
{
    const std::string input_sequence = "rose_%04d.png";
    const int num_frames = 5283;
    const std::string intermediate_data_path_prefix = "intermediate_data_";
    
    prepare_temporary_data( input_sequence, num_frames, intermediate_data_path_prefix );
    
    const std::string keyframe_sequence_name=intermediate_data_path_prefix+"subsequence_last_%04d.png";
    const std::string first_keyframe_name=intermediate_data_path_prefix+"subsequence_colorshift_0000.png";
    const double keyframe_diff_threshold=8;
    const int keyframe_num=36;
    const std::string keyframe_mask_output_path="rose_keyframe_mask_";
    
    extract_difference_mask_between_keyframe(keyframe_sequence_name, first_keyframe_name, keyframe_diff_threshold, keyframe_num, keyframe_mask_output_path);
    
    cout<<"get keyframe difference masks!"<<endl;
}
