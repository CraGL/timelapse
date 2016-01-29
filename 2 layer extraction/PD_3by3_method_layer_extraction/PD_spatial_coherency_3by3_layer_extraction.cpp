#include "stdafx.h"
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp> 
#include <opencv2/imgproc/imgproc.hpp>


#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <cmath>
#include <fstream>
#include <string>
#include <list>
#include <map>
#include <stack>
#include <iomanip>


#include <Eigen/Sparse> 
#include <Eigen/Dense>
#include <Eigen/LU>


using namespace std;
using namespace cv;
using namespace Eigen;

typedef double timemap_t;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMatrixT;


unsigned char d2b( double value )
{
    if( value < 0. ) value = 0.;
    long int result = ( value * 255. + .5 );
    if( result < 0 ) result = 0;
    if( result > 255 ) result = 255;
    return (unsigned char) result;
}
 
cv::Mat toLab( const cv::Mat& bgr )
{
	using namespace cv;
	
	Mat bgr32, lab;
	bgr.convertTo(bgr32,CV_32FC3,1.0/255.0);
	return bgr32;
	//cvtColor(bgr32,lab,CV_BGR2Lab);
	//return lab;
}

cv::Mat diff( cv::Mat mat1, cv::Mat mat2 )
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
			DIFF.ptr<float>(i)[j]=sqrt( L2.ptr<float>(i)[j]+ a2.ptr<float>(i)[j]+ b2.ptr<float>(i)[j])/1.8;///376.0;///1.8;///376.0;


	Mat imgout;
	DIFF.convertTo(imgout,CV_8UC1,255.0);
	//imshow("imgout",imgout);
	//waitKey(0);
	return imgout;

}

void Compute_three_by_three_Paint_Matrix_on_just_one_component(cv::Mat img1, cv::Mat img2, std::vector< std::pair< int, int > > &differing_pixels, SparseMatrixT& M, Eigen::VectorXd& N)
{
	using namespace cv;
         

	Eigen::SparseMatrix<double,Eigen::RowMajor> A(3,3);
	Eigen::VectorXd  B(3);

    M.resize(3,3);
	N.resize(3);

	double A00=0.0;
	double A01=0.0;
	double A02=0.0;
	double A10=0.0;
	double A11=0.0;
	double A12=0.0;
	double A20=0.0;
	double A21=0.0;
	double A22=0.0;
	double B00=0.0;
	double B10=0.0;
	double B20=0.0;

	
	int n=differing_pixels.size();
    for(int index=0; index<n;index++)
	{

		int i=differing_pixels[index].first;
		int j=differing_pixels[index].second;

		Vec3d *Bi=img1.ptr<Vec3d>(i);
		Vec3d *Ai=img2.ptr<Vec3d>(i);
 

		double ui[3]={0.0};
		double vm=std::sqrt( (Ai[j][0]-Bi[j][0])*(Ai[j][0]-Bi[j][0])+ (Ai[j][1]-Bi[j][1])*(Ai[j][1]-Bi[j][1]) + (Ai[j][2]-Bi[j][2])*(Ai[j][2]-Bi[j][2]) );
		for(int k=0;k<3;k++)
			ui[k]=(Ai[j][k]-Bi[j][k])/vm;

		//double weight=1.0;
		double weight=vm*vm;

		double ci[3]={0.0};
		double b_u=Bi[j][0]*ui[0]+ Bi[j][1]*ui[1] + Bi[j][2]*ui[2];
		for(int k=0;k<3;k++)
			ci[k]=b_u*ui[k]-Bi[j][k];

		double common=ui[0]*ui[0]+ui[1]*ui[1]+ui[2]*ui[2]-2.0;

		double tempA00=(1-ui[0]*ui[0])*(1-ui[0]*ui[0]) + ui[0]*ui[0]*(ui[1]*ui[1]+ui[2]*ui[2]);
		A00+=(weight*tempA00);
		double tempA01= ui[0]*ui[1]*common;
		A01+=(weight*tempA01);
		double tempA02= ui[0]*ui[2]*common;
		A02+=(weight*tempA02);
		double tempA10= ui[0]*ui[1]*common;
		A10+=(weight*tempA10);
		double tempA11=(1-ui[1]*ui[1])*(1-ui[1]*ui[1]) + ui[1]*ui[1]*(ui[0]*ui[0]+ui[2]*ui[2]);
		A11+=(weight*tempA11);
		double tempA12= ui[1]*ui[2]*common;
		A12+=(weight*tempA12);
		double tempA20= ui[0]*ui[2]*common;
		A20+=(weight*tempA20);
		double tempA21= ui[1]*ui[2]*common;
		A21+=(weight*tempA21);
		double tempA22= (1-ui[2]*ui[2])*(1-ui[2]*ui[2]) + ui[2]*ui[2]*(ui[1]*ui[1]+ui[0]*ui[0]);
		A22+=(weight*tempA22);
		double tempB00= ci[0]*(ui[0]*ui[0]-1) + ci[1]*ui[0]*ui[1] + ci[2]*ui[0]*ui[2];
		B00+=(weight*tempB00);
		double tempB10= ci[0]*ui[0]*ui[1] + ci[1]*(ui[1]*ui[1]-1) + ci[2]*ui[1]*ui[2];
		B10+=(weight*tempB10);
		double tempB20= ci[0]*ui[0]*ui[2] + ci[1]*ui[2]*ui[1] + ci[2]*(ui[2]*ui[2]-1);
		B20+=(weight*tempB20);
 
 
	}

	A.insert(0,0)=A00;
	A.insert(0,1)=A01;
	A.insert(0,2)=A02;
	A.insert(1,0)=A10;
	A.insert(1,1)=A11;
	A.insert(1,2)=A12;
	A.insert(2,0)=A20;
	A.insert(2,1)=A21;
	A.insert(2,2)=A22;
	B(0)=B00;
	B(1)=B10;
	B(2)=B20;

  //std::ofstream file("D:\\A_Matrix.txt");
  //if (file.is_open())
  //{
  //  file << "Here is the RowMajor matrix A:\n" << A << '\n';
  // /*   file << "m" << '\n' <<  Eigen::colm(_M) << '\n';*/
  //}
  //file.close();

  //std::ofstream file2("D:\\B_Vector.txt");
  //if (file2.is_open())
  //{
  //  file2 << "Here is the matrix B:\n" << B << '\n';
  //  /*   file << "m" << '\n' <<  Eigen::colm(_M) << '\n';*/
  //}

  //file2.close();

	M=A.transpose()*A;
	N=A.transpose()*B;

}

 
void Repair_Paint_Alpha(cv::Mat img1, cv::Mat img2, std::vector< std::pair< int, int > > &differing_pixels, std::vector< std::pair< int, int > > &sub_differing_pixels, cv::Mat &T_frame, timemap_t currentT,  Eigen::VectorXd X,cv::Mat &Paint,cv::Mat &Alpha)
{
    std::vector< std::pair< int, int > > rest_differing_pixels;
	int n=differing_pixels.size();
    for(int index=0; index<n;index++)
	{
		int i=differing_pixels[index].first;
		int j=differing_pixels[index].second;

		Vec3d *Bi=img1.ptr<Vec3d>(i);
		Vec3d *Ai=img2.ptr<Vec3d>(i);
		Vec3b *Pi=Paint.ptr<Vec3b>(i);
		Vec3b *Alphai=Alpha.ptr<Vec3b>(i);


#define Reprojection 0
#if Reprojection==0

		double u_pb[3]={0.0};
		double vm_pb=std::sqrt( (X[0]-Bi[j][0])*(X[0]-Bi[j][0])+ (X[1]-Bi[j][1])*(X[1]-Bi[j][1]) + (X[2]-Bi[j][2])*(X[2]-Bi[j][2]) );
		for(int k=0;k<3;k++)
			u_pb[k]=(X[k]-Bi[j][k])/vm_pb;

		double _a[3]={0.0};
		double t_pb=0.0;
		
		for(int k=0;k<3;k++)
		{
			t_pb+=(Ai[j][k]-Bi[j][k])*u_pb[k];
		}
		
		for(int k=0;k<3;k++)
		{
			_a[k]=Bi[j][k]+t_pb*u_pb[k];
		}



		double a_new[3]={0.0};

		double r=1.0;
		 
		double e=1.0/512;


		for(int k=0;k<3;k++)
		{
			if((_a[k]-Ai[j][k])>0)
				if(r>=e/(_a[k]-Ai[j][k]))
					r=e/(_a[k]-Ai[j][k]);
			if((_a[k]-Ai[j][k])<0)
				if(r>=-e/(_a[k]-Ai[j][k]))
					r=-e/(_a[k]-Ai[j][k]);
		}

		assert((r>0)&&(r<=1));



		for(int k=0;k<3;k++)
			a_new[k]=Ai[j][k]+r*(_a[k]-Ai[j][k]);


#elif Reprojection==1
		double a_new[3]={0.0};
		for(int k=0;k<3;k++)
			a_new[k]=Ai[j][k];
#endif

		//here, we get new a: a_new[], then we will use a_new[] to replace old Ai[j][k] to do projection.



		double ui[3]={0.0};
		double vm=std::sqrt( (a_new[0]-Bi[j][0])*(a_new[0]-Bi[j][0])+ (a_new[1]-Bi[j][1])*(a_new[1]-Bi[j][1]) + (a_new[2]-Bi[j][2])*(a_new[2]-Bi[j][2]) );
		for(int k=0;k<3;k++)
			ui[k]=(a_new[k]-Bi[j][k])/vm;




		double t=0.0;
		for(int k=0;k<3;k++)
		{
			t+=(X[k]-Bi[j][k])*ui[k];
		}

		if(t<0.0)
			t=abs(t);


		//{
		//	for(int k=0;k<3;k++)
		//	{
		//		X[k]=2*Bi[j][k]-X[k];
		//	}
		//}

		double a_b=0.0;
		for(int k=0;k<3;k++)
		{
			a_b+=(a_new[k]-Bi[j][k])*ui[k];
		}




		double l_b=1000;

		for(int k=0;k<3;k++)
		{
			if((1-Bi[j][k])/ui[k]>0)
				if(l_b>=(1-Bi[j][k])/ui[k])
					l_b=(1-Bi[j][k])/ui[k];
			if((0-Bi[j][k])/ui[k]>0)
				if(l_b>=(0-Bi[j][k])/ui[k])
					l_b=(0-Bi[j][k])/ui[k];
		}
 
		if(l_b>sqrt(3.0))
		{
			cout.precision(15);
			cout<<"error0: "<<fixed<<l_b<<endl;
			getchar();
		}

		assert(a_b<=l_b);

		if(a_b>l_b)
		{
			cout<<"error1"<<endl;
			getchar();
		}


		if(t<(a_b-sqrt(3.0)/512.0))
		{
			cout<<"error2  ";
			cout<<t<<"  "<<a_b;
			sub_differing_pixels.push_back(differing_pixels[index]);
			continue;

			//getchar();
		}
		else if(((a_b-t)<sqrt(3.0)/512)&&(t<a_b))
			t=a_b;

		if(t>(l_b+sqrt(3.0)/512.0))
		{
			cout<<"error3"<<endl;
			sub_differing_pixels.push_back(differing_pixels[index]);
			continue;
			//getchar();
		}
		else if(((t-l_b)<sqrt(3.0)/512)&&(t>l_b))
			t=l_b;




		double alpha=a_b/t;

		if(alpha>1.0)
		{
			alpha=1.0;
			cout<<"error4: "<<alpha<<endl;
			getchar();
		}
		if(alpha<(a_b/l_b))
		{
			alpha=a_b/l_b;
			cout<<"error5"<<endl;
			getchar();
		
		}


		rest_differing_pixels.push_back(differing_pixels[index]);

		for(int k=0;k<3;k++)
		{
			double p=(Bi[j][k]+t*ui[k])*255;
			Pi[j][k]=p>255? 255:p;
			Alphai[j][k]=(alpha)*255;
		}

		T_frame.ptr<timemap_t>(i)[j]=T_frame.ptr<timemap_t>(i)[j]*(1-alpha) + alpha*currentT;
	

	}

	differing_pixels.~vector();
	cout<<differing_pixels.size()<<endl;
	//differing_pixels=rest_differing_pixels;
	//cout<<differing_pixels.size()<<endl;
	cout<<sub_differing_pixels.size()<<endl;

}



/// old name is: Update_by_AlphaTime_with_Laplacian_constraints_on_just_one_component()
void PD_spatial_coherency_closet_paint_3by3_layer_extraction(cv::Mat img1, cv::Mat img2, cv::Mat &T_frame, timemap_t currentT, cv::Mat Diff_BinImage, std::vector< std::pair< int, int > > &differing_pixels,cv::Mat &Paint,cv::Mat &Alpha, int Myflag)
{
	using namespace cv;
	int width=img1.cols;
	int height=img1.rows;

	Eigen::SparseMatrix<double,Eigen::RowMajor> M;
	Eigen::VectorXd N;


	Compute_three_by_three_Paint_Matrix_on_just_one_component(img1,img2,differing_pixels,M,N);
	
	Eigen::VectorXd  X(3);
 
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(M);
	X=ldlt.solve(N);

	cout<<"X: "<<(int)d2b(X[0])<<" "<<(int)d2b(X[1])<<" "<<(int)d2b(X[2])<<endl;
	//getchar();

	//cout<<"size"<<differing_pixels.size()<<endl;

	if(ldlt.info()==Eigen::Success)
	{
		std::cout<<"success"<<std::endl;
		cout<<differing_pixels.size();

		 std::vector< std::pair< int, int > > sub_differing_pixels;

	    Repair_Paint_Alpha(img1,img2,differing_pixels,sub_differing_pixels,T_frame,currentT,X,Paint,Alpha);
		

		if(sub_differing_pixels.size()>0)
		{
			Compute_three_by_three_Paint_Matrix_on_just_one_component(img1,img2,sub_differing_pixels,M,N);
			Eigen::VectorXd  Y(3);
 
			Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(M);
			Y=ldlt.solve(N);

			cout<<"Y: "<<(int)d2b(Y[0])<<" "<<(int)d2b(Y[1])<<" "<<(int)d2b(Y[2])<<endl;
			getchar();

			std::vector< std::pair< int, int > > sub_sub_differing_pixels;

			Repair_Paint_Alpha(img1,img2,sub_differing_pixels,sub_sub_differing_pixels,T_frame,currentT,Y,Paint,Alpha);

			if(sub_sub_differing_pixels.size()>0) // means still can not get proper result, then use magic!
			{
				cout<<"Magic"<<endl;
				getchar();
				for(int d=0;d<sub_sub_differing_pixels.size();d++)
				{
					int i=sub_sub_differing_pixels[d].first;
					int j=sub_sub_differing_pixels[d].second;

					Vec3d * Bi=img1.ptr<Vec3d>(i);
					Vec3d * Ai=img2.ptr<Vec3d>(i);
					Vec3b * Pi=Paint.ptr<Vec3b>(i);
					Vec3b * Ui=Alpha.ptr<Vec3b>(i);

					double alpha=0.0;
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
 
					}

					//cout<<alpha_list.size()<<endl;

					std::sort(alpha_list.begin(),alpha_list.end());

					alpha=alpha_list[alpha_list.size()-1];// find max value

					if(Myflag ==0)
						T_frame.ptr<timemap_t>(i)[j]=T_frame.ptr<timemap_t>(i)[j]*(1-alpha) + alpha*currentT;

					for(int k=0;k<3;k++)
					{
						Pi[j][k]=((Ai[j][k]-Bi[j][k])/alpha+Bi[j][k])*255;
						Ui[j][k]=alpha*255;
					}
	 
				}
			
			}

		}

	}


	if(ldlt.info()==Eigen::NumericalIssue)
	{
		std::cout<<"fail"<<std::endl; // use magic method to replace

		for(int d=0;d<differing_pixels.size();d++)
		{
			int i=differing_pixels[d].first;
			int j=differing_pixels[d].second;

			Vec3d * Bi=img1.ptr<Vec3d>(i);
			Vec3d * Ai=img2.ptr<Vec3d>(i);
			Vec3b * Pi=Paint.ptr<Vec3b>(i);
			Vec3b * Ui=Alpha.ptr<Vec3b>(i);

			double alpha=0.0;
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
 
			}

			//cout<<alpha_list.size()<<endl;

			std::sort(alpha_list.begin(),alpha_list.end());

			alpha=alpha_list[alpha_list.size()-1];// find max value

			if(Myflag ==0)
				T_frame.ptr<timemap_t>(i)[j]=T_frame.ptr<timemap_t>(i)[j]*(1-alpha) + alpha*currentT;

			for(int k=0;k<3;k++)
			{
				Pi[j][k]=((Ai[j][k]-Bi[j][k])/alpha+Bi[j][k])*255;
				Ui[j][k]=alpha*255;
			}
	 
	 
		}


	}

 

}

 
double difference_Vec3b(Vec3b A, Vec3b B)
{
	return(sqrt(double((A[0]-B[0])*(A[0]-B[0])+(A[1]-B[1])*(A[1]-B[1])+(A[2]-B[2])*(A[2]-B[2]))));
}

//use rgb space to compute diff pixels. input is CV_8UC3
void get_diff_pixels(Mat before,Mat after,vector<std::pair<int,int>> &diff_pixels)
{
	for(int i=0;i<before.rows;i++)
	{
		Vec3b *Fi=before.ptr<Vec3b>(i);
		Vec3b *Pi=after.ptr<Vec3b>(i);
		for(int j=0;j<before.cols;j++)
		{
			double d=difference_Vec3b(Fi[j],Pi[j]);
			if(d>0.0)
			{
				diff_pixels.push_back(std::make_pair(i,j));
			}
		
		}
	}
}
 
 

int main()
{

  Mat before=imread("test_case/before.png",1);
  Mat after= imread("test_case/after.png",1);

  Mat T_frame=Mat::zeros(before.cols,before.rows,CV_64FC1);

  //compute with just one component
  std::vector<std::pair<int,int>> differing_pixels;
  Mat Paint=Mat::zeros(before.size(),CV_8UC3);
  Mat Alpha=Mat::zeros(before.size(),CV_8UC3);

  Mat before_double;
  Mat after_double;

  before.convertTo(before_double,CV_64FC3,1.0/255.0);
  after.convertTo(after_double,CV_64FC3,1.0/255.0);

  get_diff_pixels(before,after,differing_pixels);

  Mat Diff_BinImage;
  PD_spatial_coherency_closet_paint_3by3_layer_extraction(before_double,after_double,T_frame,0.0,Diff_BinImage,differing_pixels,Paint,Alpha,0);

  imwrite("test_case/paint.png",Paint);
  imwrite("test_case/alpha.png",Alpha);

}
