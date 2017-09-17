#include "stdafx.h"
#include "CrackDetect.h"


CCrackDetect::CCrackDetect()
{
	m_PI = 3.1416;
	m_threshold = 100 / 255.0;
}


CCrackDetect::~CCrackDetect()
{
}

/*************************************************************************
* Function£º
*   crackptsCompute()
* Parameters:
*   Mat& inputImg                -the input ROI area in image or the whole image
*   VecPts pts_seq               -input points
*   Mat outputFlag               -the output flag matrix, 1 denots the crack pixel
*   StructParams lineParams      -output parameters of crack
* Return:
*   void
*
* This function produces the desired crack curve. Note that the oript and the endpt is
* very important for the proposed algorithm, so make sure the marked two points are right
* and the curve between the two points is nearly straight.
************************************************************************/
void CCrackDetect::crackptsCompute(const Mat& inputImg, VecPts& pts_seq, Mat& outputFlag, StructParams& lineParams)
{
	if (inputImg.empty())
	{
		return;
	}
	Mat img = inputImg.clone();
	if (img.depth() == CV_8U)
	{
		img.convertTo(img, CV_32F);
	}
	img *= 1.0 / 255;
	m_rows = img.rows;
	m_cols = img.cols;

	bilateralFilter(img, m_Img_edge, 4, 50, 50);
	log_normalization(m_Img_edge);
	Mat Ix, Iy;
	gaussGradient(m_Img_edge, Ix, Iy);
	m_Img_gradient = abs(Ix) + abs(Iy);
	calculateDerection(m_Img_edge, 2.0);
	
	//corret points marked by user
	correctPts(pts_seq);

	m_finalFlag = Mat::zeros(m_rows, m_cols, CV_8U);
	float lineLen = 0.0;
	for (int i = 0; i < (pts_seq.size() - 1); i++)
	{
		Point oriPt = pts_seq[i];
		Point endpt = pts_seq[i + 1];
		lineLen += calcEachLine(oriPt, endpt);
	}
	outputFlag = m_finalFlag;
	lineParams.lengthLine = double(lineLen);

	Mat myWidth_mat = Mat(1, m_widthParams.size(), CV_32F, m_widthParams.data());
	Mat tmp_m, tmp_std;
	double meanV, stdV;
	meanStdDev(myWidth_mat, tmp_m, tmp_std);
	meanV = tmp_m.at<double>(0, 0);
	stdV = tmp_std.at<double>(0, 0);

	double minV, maxV;
	minMaxLoc(myWidth_mat, &minV, &maxV);

	Mat sortInd;
	sortIdx(myWidth_mat, sortInd, SORT_EVERY_ROW + SORT_DESCENDING);
	int medianInd = sortInd.at<int>(0, int(myWidth_mat.cols / 2));
	double medianV = myWidth_mat.at<float>(0, medianInd);

	lineParams.width_avg = meanV;
	lineParams.width_max = maxV;
	lineParams.width_min = minV;
	lineParams.width_std = stdV;
	lineParams.width_median = medianV;
}

void CCrackDetect::correctPts(VecPts& pts_seq)
{
	int nPts = pts_seq.size();
	if(nPts <= 1)
		return;
	Point oriPt, endPt, tempPt;
	for (int i = 0; i < (nPts - 1); i++)
	{
		oriPt = pts_seq[i];
		endPt = pts_seq[i + 1];
		tempPt = correctOriPt(oriPt, endPt);
		pts_seq[i] = tempPt;
	}
	tempPt = correctOriPt(pts_seq[nPts - 1], pts_seq[nPts - 2]);
	pts_seq[nPts - 1] = tempPt;
}

Point CCrackDetect::correctOriPt(Point oriPt, Point endPt)
{
	Point2f uq_prim = Point2f(endPt - oriPt);
	float lineLen = sqrtf(pow(uq_prim.x, 2) + pow(uq_prim.y, 2)) + 0.001;
	uq_prim = uq_prim / lineLen;
	Point2f uf_prim = Point2f(uq_prim.y, -1 * uq_prim.x);

	float maxV = 0;
	float ptV;
	Point newPt, correctPt;
	for (int ii = -10; ii <= 10; ii++)
	{
		for (int jj=0; jj <= 3; jj++)
		{
			newPt.x = round(oriPt.x + ii*uf_prim.x);
			newPt.y = round(oriPt.y + ii*uf_prim.y);
			
			newPt.x = round(newPt.x + jj*uq_prim.x);
			newPt.y = round(newPt.y + jj*uq_prim.y);
			
			if (isInbox(newPt))
			{
				ptV = calcV(newPt, uq_prim, uf_prim);
				if (ptV > maxV)
				{
					maxV = ptV;
					correctPt = newPt;
				}
			}
		}
	}
	return correctPt;
}


float CCrackDetect::calcV(Point pt, Point2f uq_prim, Point2f uf_prim)
{
	float ptV = 0;
	int ncount = 0;
	Point newPt;
	Point2f uq_newPt;
	for (int ii = -1; ii <= 1; ii++)
	{
		for (int jj = 0; jj <= 10; jj++)
		{
			newPt.x = round(oriPt.x + ii*uf_prim.x);
			newPt.y = round(oriPt.y + ii*uf_prim.y);

			newPt.x = round(newPt.x + jj*uq_prim.x);
			newPt.y = round(newPt.y + jj*uq_prim.y);
			
		    if (isInbox(newPt))
			{
				ncount++;
				uq_newPt = [uq_row(newPt(1),newPt(2)) uq_col(newPt(1),newPt(2))];
				ptV += fabs(uq_newPt.dot(uq_prim));
			}
		}
	}
	ptV = ptV/ncount;
	return ptV;
}


float CCrackDetect::calcEachLine(Point oriPt, Point endPt)
{
	Point2f uq_prim = Point2f(endPt - oriPt);
	float lineLen = sqrtf(pow(uq_prim.x, 2) + pow(uq_prim.y, 2)) + 0.001;
	uq_prim = uq_prim / lineLen;
	Point2f uf_prim = Point2f(uq_prim.y, -1 * uq_prim.x);

	float meanGrayvalue;
	meanGrayvalue = m_Img_edge.at<float>(endPt);
	trackCurve(meanGrayvalue, endPt, oriPt, endPt, uq_prim);

	meanGrayvalue = m_Img_edge.at<float>(oriPt);
	trackCurve(meanGrayvalue, oriPt, oriPt, endPt, uq_prim);

	Point currentpt;
	float currentptV;
	for (int ii = 1; ii <= floor(lineLen); ii++)
	{
		currentpt.x = round(oriPt.x + ii*uq_prim.x);
		currentpt.y = round(oriPt.y + ii*uq_prim.y);
		if (!isnotTrack(currentpt, uf_prim))
		{
			continue;
		}

		VecPts pts_seq;
		normalDerection_choose(meanGrayvalue, currentpt, uq_prim, uf_prim, pts_seq);
		if (pts_seq.empty())
		{
			normalDerection_roughExpand(meanGrayvalue, currentpt, uq_prim, uf_prim);
			continue;
		}
		else
			currentpt = clacAvgPt(pts_seq);

		Point2f uq_currentpt(uq_col.at<float>(currentpt), uq_row.at<float>(currentpt));
		currentptV = m_Img_edge.at<float>(currentpt);

		if (isSimilarity(uq_currentpt, uq_prim) && (fabs((currentptV - meanGrayvalue)) < m_threshold))
		{
			trackCurve(meanGrayvalue, currentpt, oriPt, endPt, uq_prim);
		}
		else
		{
			setFlagmatrix(pts_seq);
		}
		
		currentpt.x = round(oriPt.x + ii*uq_prim.x);
		currentpt.y = round(oriPt.y + ii*uq_prim.y);
		if (isnotTrack(currentpt, uf_prim))
		{
			normalDerection_roughExpand(meanGrayvalue, currentpt, uq_prim, uf_prim);
		}
	}
	return lineLen;
}

/*************************************************************************
* Function£º
*   trackCurve()
* Parameters:
*   float& meanGrayvalue    -the gray value of potential tracked points must be arround the parameter
*   Point pt                -track from the parameter
* Return:
*   void
*
* Track by itself. This function is used to track potention crack points arrording to tangential derection
* and set the flag matrix and re-set the mean gray value.
************************************************************************/
void CCrackDetect::trackCurve(float& meanGrayvalue, Point pt, Point oriPt, Point endPt, Point2f uq_prim)
{
	StkPts seed_pts;
	seed_pts.push(pt);

	float dreturnMeanV = 0.0;
	int nCount = 0;

	Point currentpt;
	float currentptV;
	while (!seed_pts.empty())
	{
		currentpt = seed_pts.top();
		seed_pts.pop();

		currentptV = m_Img_edge.at<float>(currentpt);
		if (isIntoLinesegment(currentpt, oriPt, endPt, uq_prim) && isIntoLineneighbor(currentpt, oriPt, endPt, uq_prim)
			&& (fabs(currentptV - meanGrayvalue) < m_threshold))
		{
			nCount += 1;
			dreturnMeanV += currentptV;
		}
		else
			continue;

		Point2f uq_currentpt(uq_col.at<float>(currentpt), uq_row.at<float>(currentpt));
		Point2f uf_currentpt(uf_col.at<float>(currentpt), uf_row.at<float>(currentpt));

		VecPts pts_seq;
		scanNormalderection(currentpt, pts_seq);
		Point2f pt_avg = setFlagmatrix(pts_seq);

		bool isPush = false;
		Point newPt;
		newPt = pt_avg + uq_currentpt;

		if (isInbox(newPt) && m_finalFlag.at<unsigned char>(newPt) == 0)
		{
			Point2f uq_newpt(uq_col.at<float>(newPt), uq_row.at<float>(newPt));
			if (isSimilarity(uq_currentpt, uq_newpt))
			{
				isPush = true;
				seed_pts.push(newPt);
			}
			else
			{
				Point newNextPt;
				newNextPt = pt_avg + 2 * uq_currentpt;
				if (isInbox(newNextPt) && m_finalFlag.at<unsigned char>(newNextPt) == 0)
				{
					Point2f uq_newNextpt(uq_col.at<float>(newNextPt), uq_row.at<float>(newNextPt));
					if (isSimilarity(uq_currentpt, uq_newNextpt))
					{
						isPush = true;
						seed_pts.push(newNextPt);
					}
				}
			}
		}
		/*if (isPush == false)
		{
			normalDerection_choose(meanGrayvalue, newPt, uq_currentpt, uf_currentpt, pts_seq, 1);
			if (pts_seq.size() >= 2)
			{			
				currentpt = clacAvgPt(pts_seq);
				seed_pts.push(currentpt);
			}
		}*/

		isPush = false;
		newPt = pt_avg - uq_currentpt;
		if (isInbox(newPt) && m_finalFlag.at<unsigned char>(newPt) == 0)
		{
			Point2f uq_newpt(uq_col.at<float>(newPt), uq_row.at<float>(newPt));
			if (isSimilarity(uq_currentpt, uq_newpt))
			{
				isPush = true;
				seed_pts.push(newPt);
			}
			else
			{
				Point newNextPt;
				newNextPt = pt_avg - 2 * uq_currentpt;
				if (isInbox(newNextPt) && m_finalFlag.at<unsigned char>(newNextPt) == 0)
				{
					Point2f uq_newNextpt(uq_col.at<float>(newNextPt), uq_row.at<float>(newNextPt));
					if (isSimilarity(uq_currentpt, uq_newNextpt))
					{
						isPush = true;
						seed_pts.push(newNextPt);
					}
				}
			}
		}
		/*if (isPush == false)
		{
			normalDerection_choose(meanGrayvalue, newPt, uq_currentpt, uf_currentpt, pts_seq, 1);
			if (pts_seq.size() >= 2)
			{
				currentpt = clacAvgPt(pts_seq);
				seed_pts.push(currentpt);
			}
		}*/
	}
	dreturnMeanV = dreturnMeanV / nCount;
	meanGrayvalue = dreturnMeanV;
}

bool CCrackDetect::isIntoLinesegment(Point pt, Point oriPt, Point endPt, Point2f uq_prim)
{
	Point2f uq_oriPt = Point2f(pt - oriPt);
	float lineLen = sqrtf(pow(uq_oriPt.x, 2) + pow(uq_oriPt.y, 2)) + 0.001;
	uq_oriPt = uq_oriPt / lineLen;

	Point2f uq_endPt = Point2f(endPt - pt);
	lineLen = sqrtf(pow(uq_endPt.x, 2) + pow(uq_endPt.y, 2)) + 0.001;
	uq_endPt = uq_endPt / lineLen;

	bool returnFlag = false;
	if (uq_oriPt.dot(uq_prim) >= 0 && uq_endPt.dot(uq_prim) >= 0)
		returnFlag = true;
	return returnFlag;
}

bool CCrackDetect::isIntoLineneighbor(Point pt, Point oriPt, Point endPt, Point2f uq_prim)
{
	Point2f uq_oriPt = Point2f(pt - oriPt);
	float lineLen = sqrtf(pow(uq_oriPt.x, 2) + pow(uq_oriPt.y, 2)) + 0.001;
	uq_oriPt = uq_oriPt / lineLen;

	float msim = uq_oriPt.dot(uq_prim);
	float normLen = lineLen * sqrtf(1 - msim*msim);

	bool returnFlag = true;
	if (normLen > 30)
		returnFlag = false;
	return returnFlag;
}

/*************************************************************************
* Function£º
*   normalDerection_choose()
* Parameters:
*   float& meanGrayvalue    -the gray value of potential tracked points must be arround the parameter
*   Point currentpt         -expand from the parameter
*   Point2f uq_prim         -the prim tangential derection
*   Point2f uf_prim         -the prim normal derection
*   VecPts& pts_seq         -points sequence
* Return:
*   void
*
* This function is used to expand potential crack points along prim normal derection
* and also store patential crack points, when the current point is not tracked by trackCurve().
************************************************************************/
void CCrackDetect::normalDerection_choose(float meanGrayvalue, Point currentpt, Point2f uq_prim, Point2f uf_prim, VecPts& pts_seq, int chooseType)
{
	if (!pts_seq.empty())
		pts_seq.clear();

	Point newPt;
	float ptV;
	if (chooseType == 0)
	{
		for (int ii = -30; ii <= 30; ii++)
		{
			newPt.x = round(currentpt.x + ii*uf_prim.x);
			newPt.y = round(currentpt.y + ii*uf_prim.y);
			if (isInbox(newPt))
			{
				Point2f uq_newpt(uq_col.at<float>(newPt), uq_row.at<float>(newPt));
				ptV = m_Img_edge.at<float>(newPt);

				if ((!isnotTrack(newPt, uq_prim)) && isSimilarity(uq_newpt, uq_prim) && (fabs((ptV - meanGrayvalue)) < m_threshold))
				{
					pts_seq.push_back(newPt);
				}
			}
		}
	}
	else
	{
		for (int ii = -2; ii <= 2; ii++)
		{
			newPt.x = round(currentpt.x + ii*uf_prim.x);
			newPt.y = round(currentpt.y + ii*uf_prim.y);
			if (isInbox(newPt))
			{
				Point2f uq_newpt(uq_col.at<float>(newPt), uq_row.at<float>(newPt));
				ptV = m_Img_edge.at<float>(newPt);
				if ((m_finalFlag.at<unsigned char>(newPt) == 0) && isSimilarity(uq_newpt, uq_prim) && (fabs((ptV - meanGrayvalue)) < m_threshold))
				{
					pts_seq.push_back(newPt);
				}
			}
		}
	}
}

void CCrackDetect::normalDerection_roughExpand(float meanGrayvalue, Point currentpt, Point2f uq_prim, Point2f uf_prim)
{
	Point newPt;
	float ptV;
	for (int ii = 0; ii <= 2; ii++)
	{
		newPt.x = round(currentpt.x + ii*uf_prim.x);
		newPt.y = round(currentpt.y + ii*uf_prim.y);
		if (isInbox(newPt))
		{
			Point2f uq_newpt(uq_col.at<float>(newPt), uq_row.at<float>(newPt));
			ptV = m_Img_edge.at<float>(newPt);
			if (isSimilarity(uq_newpt, uq_prim) && (fabs((ptV - meanGrayvalue)) < m_threshold))
			{
				m_finalFlag.at<unsigned char>(newPt) = 1;
			}
		}
		else
			break;
	}
	for (int ii = 1; ii <= 2; ii++)
	{
		newPt.x = round(currentpt.x - ii*uf_prim.x);
		newPt.y = round(currentpt.y - ii*uf_prim.y);
		if (isInbox(newPt))
		{
			Point2f uq_newpt(uq_col.at<float>(newPt), uq_row.at<float>(newPt));
			ptV = m_Img_edge.at<float>(newPt);
			if (isSimilarity(uq_newpt, uq_prim) && (fabs((ptV - meanGrayvalue)) < m_threshold))
			{
				m_finalFlag.at<unsigned char>(newPt) = 1;
			}
		}
		else
			break;
	}
}

/*************************************************************************
* Function£º
*   scanNormalderection()
* Parameters:
*   VecPts& pts_seq    -the potential tracked points
*   Point currentpt    -scan along the point's normal derection
* Return:
*   void
*
* This function scans the edge of crack curve by the information of gradient and derection
* similarity. It also can compute the width of crack.
************************************************************************/
void CCrackDetect::scanNormalderection(Point currentpt, VecPts& pts_seq)
{
	Point newPt;
	float msim;
	if (!pts_seq.empty())
		pts_seq.clear();
	pts_seq.push_back(currentpt);

	Point2f uq_currentpt(uq_col.at<float>(currentpt), uq_row.at<float>(currentpt));
	Point2f uf_currentpt(uf_col.at<float>(currentpt), uf_row.at<float>(currentpt));

	vector<float> myGradient;
	if (!myGradient.empty())
		myGradient.clear();
	myGradient.push_back(m_Img_gradient.at<float>(currentpt));
	for (int ii = 1; ii <= 10; ii++)
	{
		newPt.x = round(currentpt.x + ii*uf_currentpt.x);
		newPt.y = round(currentpt.y + ii*uf_currentpt.y);
		if (isInbox(newPt))
		{
			Point2f uq_newpt(uq_col.at<float>(newPt), uq_row.at<float>(newPt));
			msim = fabs(uq_currentpt.dot(uq_newpt)) * m_Img_gradient.at<float>(newPt);
			myGradient.push_back(msim);
		}
		else
			break;
	}
	Mat myGradient_mat = Mat(1, myGradient.size(), CV_32F, myGradient.data());
	Mat sortInd;
	sortIdx(myGradient_mat, sortInd, SORT_EVERY_ROW + SORT_DESCENDING);
	int maxInd = sortInd.at<int>(0, 0);
	for (int ii = 1; ii <= maxInd; ii++)
	{
		newPt.x = round(currentpt.x + ii*uf_currentpt.x);
		newPt.y = round(currentpt.y + ii*uf_currentpt.y);
		pts_seq.push_back(newPt);
	}
	Point edegePt1 = newPt;

	if (!myGradient.empty())
		myGradient.clear();
	myGradient.push_back(m_Img_gradient.at<float>(currentpt));
	for (int ii = 1; ii <= 10; ii++)
	{
		newPt.x = round(currentpt.x - ii*uf_currentpt.x);
		newPt.y = round(currentpt.y - ii*uf_currentpt.y);
		if (isInbox(newPt))
		{
			Point2f uq_newpt(uq_col.at<float>(newPt), uq_row.at<float>(newPt));
			msim = fabs(uq_currentpt.dot(uq_newpt)) * m_Img_gradient.at<float>(newPt);
			myGradient.push_back(msim);
		}
		else
			break;
	}
	Mat myGradient_mat2 = Mat(1, myGradient.size(), CV_32F, myGradient.data());
	Mat sortInd2;
	sortIdx(myGradient_mat2, sortInd2, SORT_EVERY_ROW + SORT_DESCENDING);
	maxInd = sortInd2.at<int>(0, 0);
	for (int ii = 1; ii <= maxInd; ii++)
	{
		newPt.x = round(currentpt.x - ii*uf_currentpt.x);
		newPt.y = round(currentpt.y - ii*uf_currentpt.y);
		pts_seq.push_back(newPt);
	}
	Point edegePt2 = newPt;
	Point2f lenPt = Point2f(edegePt1 - edegePt2);
	float lineLen = sqrtf(pow(lenPt.x, 2) + pow(lenPt.y, 2));
	m_widthParams.push_back(lineLen);
}

//calculate and judge the derection similarity
bool CCrackDetect::isSimilarity(Point2f u_currentPt, Point2f u_newPt)
{
	bool returnFlag;
	float msim = u_currentPt.dot(u_newPt);
	if (fabs(msim) >= 0.9)
		returnFlag = true;
	else
		returnFlag = false;
	return returnFlag;
}

//set potential crack points to 1
Point2f CCrackDetect::setFlagmatrix(VecPts pts_seq)
{
	Point tempPt(0, 0);
	int i;
	for (i = 0; i < pts_seq.size(); i++)
	{
		Point pt = pts_seq[i];
		tempPt += pt;
		m_finalFlag.at<unsigned char>(pt) = 1;
	}
	Point2f returnPt = Point2f(tempPt) / i;
	return returnPt;
}

Point CCrackDetect::clacAvgPt(VecPts pts_seq)
{
	/*Point tempPt(0, 0);
	int i;
	for (i = 0; i < pts_seq.size(); i++)
	{
		Point pt = pts_seq[i];
		tempPt += pt;
	}
	Point2f returnPt = Point2f(tempPt) / i;
	return Point(returnPt);*/

	int idx = pts_seq.size();
	int medianIdx = floor(idx / 2.0);
	Point pt = pts_seq[medianIdx];
	return pt;
}

bool CCrackDetect::isInbox(Point pt, Mat& Img)
{
	bool returnFlag;
	if (pt.x >= 0 && pt.x < Img.cols && pt.y >= 0 && pt.y < Img.rows)
		returnFlag = true;
	else
		returnFlag = false;
	return returnFlag;
}

bool CCrackDetect::isInbox(Point pt)
{
	bool returnFlag = false;
	if (pt.x >= 0 && pt.x < m_cols && pt.y >= 0 && pt.y < m_rows)
		returnFlag = true;
	return returnFlag;
}

//judge current point is tracked or not
bool CCrackDetect::isnotTrack(Point pt, Point2f uf_prim, int thr)
{
	bool returnFlag = true;
	for (int i = 0; i <= thr; i++)
	{
		Point newPt;
		newPt.x = round(pt.x + i*uf_prim.x);
		newPt.y = round(pt.y + i*uf_prim.y);
		if (isInbox(newPt))
		{
			if (m_finalFlag.at<unsigned char>(newPt) == 1)
			{
				returnFlag = false;
				break;
			}
		}
		else
			break;
	}
	if (returnFlag == false)
		return returnFlag;
	for (int i = 1; i <= thr; i++)
	{
		Point newPt;
		newPt.x = round(pt.x - i*uf_prim.x);
		newPt.y = round(pt.y - i*uf_prim.y);
		if (isInbox(newPt))
		{
			if (m_finalFlag.at<unsigned char>(newPt) == 1)
			{
				returnFlag = false;
				break;
			}
		}
		else
			break;
	}
	return returnFlag;
}

void CCrackDetect::log_normalization(Mat& Img)
{
	Img *= 255.0;
	Img = (Img + 1)*0.5;
	Mat logImg;
	log(Img, logImg);

	Mat tmp_m, tmp_std;
	double meanV, stdV;
	meanStdDev(logImg, tmp_m, tmp_std);
	meanV = tmp_m.at<double>(0, 0);
	stdV = tmp_std.at<double>(0, 0);

	logImg = (logImg - meanV) / stdV;
	double minV, maxV;
	minMaxLoc(logImg, &minV, &maxV);

	Img = (logImg - minV) / (maxV - minV);
}

//gaussian function
float CCrackDetect::gaussCompute(int xV, float gaussSigma)
{
	//y = exp(-x^2/(2*sigma^2)) / (sigma*sqrt(2*pi));
	float returnV;
	returnV = exp(-xV*xV / (2.0*gaussSigma*gaussSigma)) / (gaussSigma*sqrtf(2 * m_PI));
	return returnV;
}

//first-order gaussian function
float CCrackDetect::dgaussCompute(int xV, float gaussSigma)
{
	//y = -x * gauss(x,sigma) / sigma^2;
	float returnV;
	returnV = -xV * gaussCompute(xV, gaussSigma) / (gaussSigma*gaussSigma);
	return returnV;
}

/*************************************************************************
* Function£º
*   calculateDerection()
* Parameters:
*   Mat& IM             -input image
*   Mat& IX             -output gradient image along x
*   Mat& IY             -output gradient image along y
*   float gaussSigma    -the standard deviation of gaussian function
* Return:
*   void
*
* This function produces gaussian gradient image. Generate first-order gaussian kernel map,
* and then brifely convolve the input image.
************************************************************************/
void CCrackDetect::gaussGradient(const Mat& IM, Mat& IX, Mat& IY, float gaussSigma)
{
	float epsilon = 1e-2;
	int win_halfsize = ceil(gaussSigma*sqrt(-2 * log(sqrt(2 * m_PI)*gaussSigma*epsilon)));
	int win_size = 2 * win_halfsize + 1;

	Mat kernelHX = Mat(win_size, win_size, CV_32F);
	float* dataptr = (float*)kernelHX.data;
	for (int i = 0; i < win_size; i++)
	{
		for (int j = 0; j < win_size; j++)
		{
			*dataptr++ = gaussCompute(i - win_halfsize, gaussSigma) * dgaussCompute(j - win_halfsize, gaussSigma);
		}
	}

	kernelHX = kernelHX / (sqrtf(sum(kernelHX.mul(kernelHX))[0]));
	Mat kernelHY = kernelHX.t();

	filter2D(IM, IX, IM.depth(), kernelHX, Point(-1, -1), 0.0, BORDER_REPLICATE);
	filter2D(IM, IY, IM.depth(), kernelHY, Point(-1, -1), 0.0, BORDER_REPLICATE);
}

/*************************************************************************
* Function£º
*   calculateDerection()
* Parameters:
*   Mat& IM             -input image
*   float gaussSigma    -the standard deviation of gaussian function
* Return:
*   void
*
* This function produces tangential derection and normal derection for each point, where
* the eigen vector of max eigen value in Hessian matrix corresponds to normal derection.
************************************************************************/
void CCrackDetect::calculateDerection(const Mat& IM, float gaussSigma)
{
	Mat Ix, Iy, Ixx, Ixy, Iyy;
	gaussGradient(IM, Ix, Iy, gaussSigma);
	gaussGradient(Ix, Ixx, Ixy, gaussSigma);
	gaussGradient(Iy, Ixy, Iyy, gaussSigma);

	Mat signMat, sqrtMat, tempMat, subMat;
	Mat addMat = Ixx + Iyy;
	calculateSign(addMat, signMat);
	subMat = Ixx - Iyy;
	tempMat = subMat.mul(subMat) + 4 * Ixy.mul(Ixy);
	sqrt(tempMat, sqrtMat);

	Mat lamda1 = 0.5 * (addMat + signMat.mul(sqrtMat));
	Mat lamda2 = (Ixx.mul(Iyy) - Ixy.mul(Ixy)) / lamda1;
	Mat lamda = max(lamda1, lamda2);

	subMat = lamda - Ixx;
	tempMat = Ixy.mul(Ixy) + subMat.mul(subMat);
	sqrt(tempMat, sqrtMat);
	uf_col = abs(Ixy) / sqrtMat;

	calculateSign(Ixy, signMat);
	uf_row = signMat.mul(subMat) / sqrtMat;

	uq_row = 1.0 * uf_col;
	uq_col = -1.0 * uf_row;
}

//sign computation for matrix
void CCrackDetect::calculateSign(const Mat& IM, Mat& outputMat)
{
	IM.copyTo(outputMat);
	float tempV;
	float* dataptr = (float*)IM.data;
	float* dataptr_output = (float*)outputMat.data;
	for (int i = 0; i < IM.rows; i++)
	{
		for (int j = 0; j < IM.cols; j++)
		{
			tempV = *dataptr++;
			if (tempV > 0)
				*dataptr_output = 1.0;
			else if (tempV == 0)
				*dataptr_output = 0.0;
			else
				*dataptr_output = -1.0;
			dataptr_output++;
		}
	}

}