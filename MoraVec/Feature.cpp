#include "stdio.h"
#include "vector"
#include "math.h"
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include<opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgproc/types_c.h>
#include<iostream>

using namespace cv;
using namespace std;


#define windowsize 9                    //得到某一点的兴趣值窗口大小
#define Threshold 500                   //计算兴趣点的阈值设定

#define matchsize 41                   //相关系数窗口大小
#define matchThreshold 0.5             //相关系数的匹配系数

//像素特征值
struct FeaturePoint {
	float value;
	int rows;
	int cols;
};

//函数取最小值
int GetMin(float* V, int num) {
	float min = V[0];
	for (int i = 0; i < num; i++) {
		if (V[i] < min) {
			min = V[i];
		}
	}
	return min;
}

//得到某一点的兴趣值
int GetIV(Mat M, int r, int c) {
	int W = 0;
	W = int(windowsize/ 2);    //W = 4
	float Feature = 0;
	

	float V[4] = { 0,0,0,0 };

	for (int i = 0; i < W ; i++) {
		V[0] += pow(M.at<uchar>(r + i, c) - M.at<uchar>(r + i + 1, c), 2);

		V[1] += pow(M.at<uchar>(r, c + i) - M.at<uchar>(r, c + i + 1), 2);

		V[2] += pow(M.at<uchar>(r + i, c + i) - M.at<uchar>(r + i + 1, c + i + 1), 2);

		V[3] += pow(M.at<uchar>(r + i, c - i) - M.at<uchar>(r + i + 1, c - i - 1), 2);
	}

	Feature = GetMin(V, 4);

	return Feature;    
}
//返回极小值，即该像素点的兴趣值

//特征点提取
void GetFeature(Mat M, vector<Point3f>& featureM1) {

	if (M.empty()) {
		cout << "Fail to read the image!!!" << endl;
		/*return -1;*/
	}
	/*namedWindow("原图片");
	imshow("原图片", M);*/

	Mat imgGray;
	cvtColor(M, imgGray, COLOR_RGB2GRAY);//转变为灰度图像空间	
	GaussianBlur(imgGray, imgGray, Size(5, 5), 0, 0);//使用高斯滤波预处理
	Mat MoraVecFeature;
	MoraVecFeature = Mat::zeros(imgGray.size(), CV_32FC1);

	//定义一些必要变量
	int cols = imgGray.cols;
	int rows = imgGray.rows;
	int W = int(windowsize / 2);
	int k = 0;
	int num = 0;
	FeaturePoint* point1 = (FeaturePoint*)malloc(sizeof(FeaturePoint) * 20000);
	
	for (int i = W ; i < imgGray.rows - W - 1; i++)
	{
		for (int j = W ; j < imgGray.cols - W - 1; j++)
		{
			float min = GetIV(imgGray, i, j);
			if (GetIV(imgGray, i, j) < Threshold) {
				MoraVecFeature.at<float>(i, j) = 0;
			}
			else {
				MoraVecFeature.at<float>(i, j) = min;
			}
				
		}
	}

	//将候选点中兴趣值不是最大的去掉，只留一个兴趣值最大者
	for (int i = W; i < MoraVecFeature.rows - 1 - W; i++)
	{
		for (int j = W; j < MoraVecFeature.cols - 1 - W; j++)
		{
			float t1 = MoraVecFeature.at<float>(i, j);
			for (int m = 0; m < windowsize; m++)
			{
				for (int n = 0; n < windowsize; n++)
				{
					float t2 = MoraVecFeature.at<float>(i - W + m, j - W + n);
					if (t1 < t2)
					{
						MoraVecFeature.at<float>(i, j) = 0;
						n = windowsize;
						m = windowsize;
					}
				}
			}
		}
	}

	//存储特征点
	for (int i = W; i < MoraVecFeature.rows - W - 1; i++)
	{
		for (int j = W; j < MoraVecFeature.cols - W - 1; j++)
		{

			if (MoraVecFeature.at<float>(i, j) > 0)
			{
				Point3f point;
				point.x = i;
				point.y = j;
				point.z = MoraVecFeature.at<float>(i, j);
				featureM1.push_back(point);
			}
		}
	}


	//将特征点用圆形表示
	int r = 5;                               //半径设为5
	for (int n = 0; n < featureM1.size(); n++)
	{
		int i = int(featureM1.at(n).y);
		int j = int(featureM1.at(n).x);
		circle(M, Point(i, j), r, Scalar(255, 255, 0), 1, 4, 0);
		line(M, Point(i - r - 2, j), Point(i + r + 2, j), Scalar(255, 0, 0), 1, 8, 0);
		line(M, Point(i, j + r + 2), Point(i, j - r - 2), Scalar(255, 0, 0), 1, 8, 0);
	}

	imshow("提取特征点图片", M);
	waitKey(0);
}

 
//相关系数图像匹配

//计算相关系数    输入x，y。 从左上角开始计算
float GetCoefficient(Mat SearchLeftWindow,Mat imageRight, int x, int y)
{
	//根据左搜索窗口确定右搜索窗口的大小
	Mat matchRightWindow;
	matchRightWindow.create(SearchLeftWindow.rows, SearchLeftWindow.cols, CV_32FC1);


	float aveRImg = 0;
	for (int m = 0; m < SearchLeftWindow.rows; m++)
	{
		for (int n = 0; n < SearchLeftWindow.cols; n++)
		{
			aveRImg += imageRight.at<uchar>(x + m, y + n);
			matchRightWindow.at<float>(m, n) = imageRight.at<uchar>(x + m, y + n);
		}
	}
	aveRImg = aveRImg / (SearchLeftWindow.rows * SearchLeftWindow.cols);
	for (int m = 0; m < SearchLeftWindow.rows; m++)
	{
		for (int n = 0; n < SearchLeftWindow.cols; n++)
		{
			matchRightWindow.at<float>(m, n) -= aveRImg;
		}
	}
	//开始计算相关系数
	float cofficent1 = 0;
	float cofficent2 = 0;
	float cofficent3 = 0;
	for (int m = 0; m < SearchLeftWindow.rows; m++)
	{
		for (int n = 0; n < SearchLeftWindow.cols; n++)
		{
			cofficent1 += SearchLeftWindow.at<float>(m, n) * matchRightWindow.at<float>(m, n);
			cofficent2 += matchRightWindow.at<float>(m, n) * matchRightWindow.at<float>(m, n);
			cofficent3 += SearchLeftWindow.at<float>(m, n) * SearchLeftWindow.at<float>(m, n);
		}
	}
	double cofficent = cofficent1 / sqrt(cofficent2 * cofficent3);
	return cofficent;
}

//相关系数匹配
void Matching(Mat M1, Mat M2, vector<Point3f> featureM1) {
	//图片预处理
	Mat imgGrayLeft;
	cvtColor(M1, imgGrayLeft, COLOR_RGB2GRAY);//转变为灰度图像空间	
	Mat imgGrayRight;
	cvtColor(M2, imgGrayRight, COLOR_RGB2GRAY);//转变为灰度图像空间	
	
	
	//定义常量
	int half_matchsize = matchsize / 2;
	float dis_col = 852;
	vector<Point3f> featureM2;           //右图像的特征匹配点


	//删除无法提供一个完整搜索框的点
	for (int i = 0; i < featureM1.size(); i++) {
		if ((featureM1.at(i).y < half_matchsize + 1) || (imgGrayLeft.cols - featureM1.at(i).y < half_matchsize + 1)) {
			featureM1.erase(featureM1.begin() + i);
			i--;
			continue;
		}
		if ((featureM1.at(i).x < half_matchsize + 1) || (imgGrayLeft.rows - featureM1.at(i).x < half_matchsize + 1)) {
			featureM1.erase(featureM1.begin() + i);
			i--;
			continue;
		}
	}

	//创建左侧图像搜索窗口      matchsize * matchsize 大小 即 31 * 31
	Mat SearchLeftWindow;
	SearchLeftWindow.create(matchsize, matchsize, CV_32FC1);

	for (int i = 0; i < featureM1.size(); i++) {
		float average = 0;  //定义每一个搜索框中的average
		

		for (int m = 0; m < matchsize; m++) {
			for (int n = 0; n < matchsize; n++) {
					average += imgGrayLeft.at<uchar>(featureM1.at(i).x - half_matchsize + m, featureM1.at(i).y - half_matchsize + n);
					SearchLeftWindow.at<float>(m, n) = imgGrayLeft.at<uchar>(featureM1.at(i).x - half_matchsize + m, featureM1.at(i).y - half_matchsize + n);
			}
		}

		average = average / (matchsize * matchsize);     

		for (int m = 0; m < matchsize; m++) {
			for (int n = 0; n < matchsize; n++) {
				SearchLeftWindow.at<float>(m, n) = SearchLeftWindow.at<float>(m, n) - average;
			}
		}


		float searchradius = 30;         //搜索半径
		vector<Point3f>M_featureM2;   //中间对应特征点储存容器

		//开始搜索右图像
		for (int m = -searchradius; m <= searchradius; m++) {
			for (int n = -searchradius; n <= searchradius; n++) {
				if ((featureM1.at(i).x )<  (searchradius+half_matchsize+1) || (imgGrayRight.rows - featureM1.at(i).x ) < (searchradius+ half_matchsize + 1)
					|| (featureM1.at(i).y + dis_col - imgGrayLeft.cols) < (searchradius+ half_matchsize + 1) || (imgGrayRight.cols + imgGrayLeft.cols - featureM1.at(i).y - dis_col) < (searchradius+ half_matchsize + 1)) {
					Point3f SearchPoint;  //可以搜寻的点的集合
					SearchPoint.x = 0;
					SearchPoint.y = 0;
					SearchPoint.z = 0;
					M_featureM2.push_back(SearchPoint);      //将其读入中间对应特征点容器
				}
				else
				{
					Point3f SearchPointM2;
					int x = featureM1.at(i).x + m - half_matchsize;
					int y = featureM1.at(i).y + dis_col - imgGrayLeft.cols + n - half_matchsize;
					float  coffee = GetCoefficient(SearchLeftWindow, imgGrayRight, x, y);
					SearchPointM2.x = featureM1.at(i).x + m;
					SearchPointM2.y = featureM1.at(i).y + dis_col - imgGrayLeft.cols + n;
					SearchPointM2.z = coffee;
					M_featureM2.push_back(SearchPointM2);
				}
			}
		}
		for (int num = 0; num < M_featureM2.size() - 1; num++) {
			float a = 0;
			float a_x = 0;
			float a_y = 0;
			// 内层for循环控制相邻的两个元素进行比较
			for (int j = num + 1; j < M_featureM2.size(); j++) {
				if (M_featureM2.at(num).z < M_featureM2.at(j).z) {
					a = M_featureM2.at(j).z;
					M_featureM2.at(j).z = M_featureM2.at(num).z;
					M_featureM2.at(num).z = a;

					a_x = M_featureM2.at(j).x;
					M_featureM2.at(j).x = M_featureM2.at(num).x;
					M_featureM2.at(num).x = a_x;

					a_y = M_featureM2.at(j).y;
					M_featureM2.at(j).y = M_featureM2.at(num).y;
					M_featureM2.at(num).y = a_y;
				}
			}
		}
		
		if (M_featureM2.at(0).z > matchThreshold && M_featureM2.at(0).z < 1) {
			Point3f Middle;
			Middle.x = M_featureM2.at(0).x;
			Middle.y = M_featureM2.at(0).y;
			Middle.z = M_featureM2.at(0).z;
			featureM2.push_back(Middle);
		}
		else
		{
			featureM1.erase(featureM1.begin() + i);
			i--;
			continue;
		}
	}
	
	//左右图像展现在一张图上
	Mat bothview;
	bothview.create(M1.rows, M1.cols + M2.cols, M1.type());
	for (int i = 0; i < M1.rows; i++) {
		for (int j = 0; j < M1.cols; j++) {
			bothview.at<Vec3b>(i, j) = M1.at<Vec3b>(i, j);
		}
	}
	for (int i = 0; i < M2.rows; i++) {
		for (int j = M1.cols; j < M1.cols + M2.cols; j++) {
			bothview.at<Vec3b>(i, j) = M2.at<Vec3b>(i, j - M1.cols);
		}
	}

	//绘制
	for (int i = 0; i < featureM2.size(); i++) {

		int radius = 5;//描绘特征点半径
		int m1 = int(featureM1.at(i).y);
		int n1 = int(featureM1.at(i).x);
		circle(bothview, Point(m1, n1), radius, Scalar(255, 255, 0), 1, 4, 0);
		line(bothview, Point(m1 - 2 - radius, n1), Point(m1 + 2 + radius, n1), Scalar(255, 0, 0), 1, 8, 0);
		line(bothview, Point(m1, n1 - 2 - radius), Point(m1, n1 + 2 + radius), Scalar(255, 0, 0), 1, 8, 0);

		int m2 = int(featureM2.at(i).y + M1.cols);
		int n2 = int(featureM2.at(i).x);
		circle(bothview, Point(m2, n2), radius, Scalar(255, 255, 0), 1, 4, 0);
		line(bothview, Point(m2 - 2 - radius, n2), Point(m2 + 2 + radius, n2), Scalar(255, 0, 0), 1, 8, 0);
		line(bothview, Point(m2, n2 - 2 - radius), Point(m2, n2 + 2 + radius), Scalar(255, 0, 0), 1, 8, 0);

		line(bothview, Point(m1, n1), Point(m2, n2), Scalar(0, 255, 255), 1, 8, 0);

	}

	Mat bothview1;
	resize(bothview, bothview1, Size(0, 0), 0.75, 0.75, INTER_LINEAR);
	imshow("相关系数匹配", bothview1);
	waitKey(0);

}


int main()
{
	Mat M1 = imread("u0369_panLeft.bmp");
	Mat M2 = imread("u0367_panRight.bmp");

	if (M1.empty() ) {
		printf("Can't open image!");
		return -1;
	}
	if (M2.empty()) {
		printf("Can't open image!");
		return -1;
	}
	
	Mat M3 = M1;
	vector<Point3f> featureM1;                  //定义左图像特征点向量
	vector<Point3f> featureM2;

	GetFeature(M1, featureM1);

	Matching(M1, M2, featureM1);

	/*GetFeature(M2, featureM2);
	int a = featureM1.at(1).x;
	int b = featureM1.at(1).y;
	int c = featureM2.at(1).x;
	int d = featureM2.at(1).y;
	printf("featureM1.at(0).x = %d\n featureM1.at(0).y = %d\n featureM2.at(0).x = %d\n   featureM2.at(0).y = %d\n", a, b, c, d);*/



	//Mat imgGrayLeft;
	//cvtColor(M1, imgGrayLeft, COLOR_RGB2GRAY);//转变为灰度图像空间	
	//Mat imgGrayRight;
	//cvtColor(M2, imgGrayRight, COLOR_RGB2GRAY);//转变为灰度图像空间	

	//float a, b, c,d;
	//a = imgGrayLeft.cols;
	//b = imgGrayRight.cols;
	//c = imgGrayLeft.rows;
	//d = imgGrayRight.rows;
	//printf("imgGrayLeft.cols = %f, imgGrayRight.cols = %f,imgGrayLeft.rows = %f,imgGrayRight.rows; = %f", a, b,c,d);
	return 0;
}

