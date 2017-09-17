#pragma once
/* crack inspection
The main principle is that the neighbor points in curve have similar tangential
direction.
The proposed algorithm contains two main steps : track curve self according to
tangential direction, and expand roughly according to normal direction

Contributed by Andrew, and written at BJTU, Beijing, China. */

#include <opencv.hpp>
#include <stack>
#include <math.h>

using namespace std;
using namespace cv;

typedef stack<Point> StkPts;
typedef vector<Point> VecPts;
typedef vector<float> VecWidthParams;

struct StructParams
{
	double width_max;
	double width_min;
	double width_avg;
	double width_median;
	double width_std;
	double lengthLine;
};

class CCrackDetect
{
public:
	CCrackDetect();
	~CCrackDetect();
	//return the flag matrix of the crack   (Mat& outputFlag)
	void crackptsCompute(const Mat& inputImg, VecPts& pts_seq, Mat& outputFlag, StructParams& lineParams);

protected:
	void trackCurve(float& meanGrayvalue, Point currentpt, Point oriPt, Point endPt, Point2f uq_prim);
	bool isIntoLinesegment(Point pt, Point oriPt, Point endPt, Point2f uq_prim);
	bool isIntoLineneighbor(Point pt, Point oriPt, Point endPt, Point2f uq_prim);
	bool isInbox(Point pt, Mat& Img);
	bool isInbox(Point pt);
	void scanNormalderection(Point currentpt, VecPts& pts_seq);
	Point2f setFlagmatrix(VecPts pts_seq);
	Point clacAvgPt(VecPts pts_seq);
	void normalDerection_choose(float meanGrayvalue, Point currentpt, Point2f uq_prim, Point2f uf_prim, VecPts& pts_seq, int chooseType = 0);
	void normalDerection_roughExpand(float meanGrayvalue, Point currentpt, Point2f uq_prim, Point2f uf_prim);
	bool isSimilarity(Point2f u_currentPt, Point2f u_newPt);
	bool isnotTrack(Point pt, Point2f uf_prim, int thr = 10);
	void log_normalization(Mat& Img);
	float gaussCompute(int xV, float gaussSigma);
	float dgaussCompute(int xV, float gaussSigma);
	void gaussGradient(const Mat& IM, Mat& IX, Mat& IY, float gaussSigma = 1.0);
	void calculateDerection(const Mat& IM, float gaussSigma = 2.0);
	void calculateSign(const Mat& IM, Mat& outputMat);
	float calcEachLine(Point oriPt, Point endpt);
	
	void correctPts(VecPts& pts_seq);
	Point correctOriPt(Point oriPt, Point endPt);
	float calcV(Point pt, Point2f uq_prim, Point2f uf_prim);

private:
	int m_rows;              //height of image
	int m_cols;              //width of image

	Mat m_Img_edge;          //bilateral filter image
	Mat m_Img_gradient;      //gradient map
	Mat m_finalFlag;         //flag map

	Mat uf_row;              //normal derection map along y
	Mat uf_col;              //normal derection map along x
	Mat uq_row;              //tangential derection map along y
	Mat uq_col;              //tangential derection map along x

	VecWidthParams m_widthParams;   //width of crack
	
	float m_PI;
	float m_threshold;
};