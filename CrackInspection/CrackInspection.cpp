// CrackInspection.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "CrackDetect.h"
#include <iostream>
#include <opencv.hpp>

using namespace std;
using namespace cv;

void calcPreRecF(Mat& result, Mat& gtruth)
{
	Mat resultM = result > 0;
	Mat gtruthM = gtruth > 0;
	Mat countNonzeroM;
	findNonZero(resultM, countNonzeroM);
	int detectNum = countNonzeroM.rows;

	findNonZero(gtruthM, countNonzeroM);
	int gtNum = countNonzeroM.rows;

	Mat interM = resultM & gtruthM;
	findNonZero(interM, countNonzeroM);
	int interNum = countNonzeroM.rows;

	float Pre = interNum / float(detectNum + 1e-3);
	float Rec = interNum / float(gtNum + 1e-3);
	float F = 2 * Pre * Rec / (Pre + Rec + 1e-3);

	cout << "Pre: " << Pre << "  Rec: " << Rec << "  F: " << F << endl;
}

int _tmain(int argc, _TCHAR* argv[])
{
	CCrackDetect CK;

	Mat img = imread("D:\\HHH.bmp", CV_LOAD_IMAGE_UNCHANGED);
	if (img.empty())
		return -1;
	if (img.channels() != 1)
	{
		cvtColor(img, img, CV_RGB2GRAY);
	}

	Mat outputflag;
	VecPts pts_seq;
	StructParams lineParams;
	//pts_seq.push_back(Point(7, 100));
	//pts_seq.push_back(Point(287, 94));
	//pts_seq.push_back(Point(468, 94));
	//pts_seq.push_back(Point(1228, 112));

	//HH
	//pts_seq.push_back(Point(24, 155));
	//pts_seq.push_back(Point(1779, 86));
	//pts_seq.push_back(Point(2972, 139));

	//HHH
	pts_seq.push_back(Point(13, 112));
	pts_seq.push_back(Point(1736, 95));
	pts_seq.push_back(Point(2895, 71));

	//pts_seq.push_back(Point(20, 112));
	//pts_seq.push_back(Point(299, 110));
	//pts_seq.push_back(Point(751, 119));

	//pts_seq.push_back(Point(9, 70));
	//pts_seq.push_back(Point(1820, 78));
	//pts_seq.push_back(Point(2761, 140));

	//pts_seq.push_back(Point(8, 167));
	//pts_seq.push_back(Point(1108, 160));
	//pts_seq.push_back(Point(1302, 147));
	//pts_seq.push_back(Point(1995, 106));
	//pts_seq.push_back(Point(2486, 88));
	//pts_seq.push_back(Point(3251, 30));
	CK.crackptsCompute(img, pts_seq, outputflag, lineParams);

	cout << "length:\n" << lineParams.lengthLine << endl;
	cout << "width_avg:" << lineParams.width_avg << "  width_max:" << lineParams.width_max << 
		"  width_min:" << lineParams.width_min << "  width_std:" << lineParams.width_std << 
		"  width_median:" << lineParams.width_median << endl;

	//Mat gtruthM = imread("D:\\HC_GT.bmp", CV_LOAD_IMAGE_UNCHANGED);
	//calcPreRecF(outputflag, gtruthM);
	

	Mat finaloutcome = Mat(img.size(), CV_8UC3);
	unsigned char* myflag = (unsigned char*)outputflag.data;
	unsigned char* myImg = (unsigned char*)img.data;
	unsigned char* myfinal;
	for (int i = 0; i < img.rows; i++)
	{
		myfinal = finaloutcome.ptr<unsigned char>(i);
		for (int j = 0; j < img.cols; j++)
		{
			if (*myflag == 1)
			{
				myfinal[3 * j] = 0;
				myfinal[3 * j + 1] = 0;
				myfinal[3 * j + 2] = 255;
			}
			else
			{
				myfinal[3 * j] = *myImg;
				myfinal[3 * j + 1] = *myImg;
				myfinal[3 * j + 2] = *myImg;
			}
			myflag++;
			myImg++;
		}
	}
	

	Mat finalcome;
	resize(finaloutcome, finalcome, Size(), 0.3, 0.3);
	imshow("image", finalcome);
	if (waitKey(0) & 0xff == 27)
	destroyAllWindows(); 
}

