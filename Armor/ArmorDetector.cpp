/**************************************************************

MIT License

Copyright (c) 2018 SEU-SuperNova-CVRA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Authors:	Rick_Hang, <213162574@seu.edu.cn>
		BinYan Hu
**************************************************************/
#include"ArmorDetector.h"
#include<iostream>
#include<vector>
#include<math.h>
#include<string>

#include"../General/opencv_extended.h"
#include"../General/numeric_rm.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

namespace rm
{

enum
{
	WIDTH_GREATER_THAN_HEIGHT,
	ANGLE_TO_UP
};
/*
*	@Brief:		// regulate the rotated rect
*	@Input:		// rotated rec
*				// regulation mode
*	@Return:	// regulated rec
*/
cv::RotatedRect& adjustRec(cv::RotatedRect& rec, const int mode)
{
	using std::swap;

	float& width = rec.size.width;
	float& height = rec.size.height;
	float& angle = rec.angle;

	if(mode == WIDTH_GREATER_THAN_HEIGHT)
	{
		if(width < height)
		{
			swap(width, height);
			angle += 90.0;
		}
	}

	while(angle >= 90.0) angle -= 180.0;
	while(angle < -90.0) angle += 180.0;

	if(mode == ANGLE_TO_UP)
	{
		if(angle >= 45.0)
		{
			swap(width, height);
			angle -= 90.0;
		}
		else if(angle < -45.0)
		{
			swap(width, height);
			angle += 90.0;
		}
	}

	return rec;
}


ArmorDescriptor::ArmorDescriptor()
{
	rotationScore = 0;
	sizeScore = 0;
	vertex.resize(4);
	for(int i = 0; i < 4; i++)
	{
		vertex[i] = cv::Point2f(0, 0);
	}
	type = UNKNOWN_ARMOR;
}

ArmorDescriptor::ArmorDescriptor(const LightDescriptor & lLight, const LightDescriptor & rLight, const int armorType, const cv::Mat & grayImg, float rotaScore, ArmorParam _param)
{
	//handle two lights
	lightPairs[0] = lLight.rec();
	lightPairs[1] = rLight.rec();

	cv::Size exLSize(int(lightPairs[0].size.width), int(lightPairs[0].size.height * 2));
	cv::Size exRSize(int(lightPairs[1].size.width), int(lightPairs[1].size.height * 2));
	cv::RotatedRect exLLight(lightPairs[0].center, exLSize, lightPairs[0].angle);
	cv::RotatedRect exRLight(lightPairs[1].center, exRSize, lightPairs[1].angle);

	cv::Point2f pts_l[4];
	exLLight.points(pts_l);
	cv::Point2f upper_l = pts_l[2];
	cv::Point2f lower_l = pts_l[3];

	cv::Point2f pts_r[4];
	exRLight.points(pts_r);
	cv::Point2f upper_r = pts_r[1];
	cv::Point2f lower_r = pts_r[0];

	vertex.resize(4);
	vertex[0] = upper_l;
	vertex[1] = upper_r;
	vertex[2] = lower_r;
	vertex[3] = lower_l;

	//set armor type
	type = armorType;

	//get front view
	getFrontImg(grayImg);
	rotationScore = rotaScore;

	// calculate the size score
	float normalized_area = contourArea(vertex) / _param.area_normalized_base;
	sizeScore = exp(normalized_area);

	// calculate the distance score
	Point2f srcImgCenter(grayImg.cols / 2, grayImg.rows / 2);
	float sightOffset = cvex::distance(srcImgCenter, cvex::crossPointOf(array<Point2f, 2>{vertex[0],vertex[2]}, array<Point2f, 2>{vertex[1],vertex[3]}));
	distScore = exp(-sightOffset / _param.sight_offset_normalized_base);

}

void ArmorDescriptor::getFrontImg(const Mat& grayImg)
{
	using cvex::distance;
	const Point2f&
		tl = vertex[0],
		tr = vertex[1],
		br = vertex[2],
		bl = vertex[3];

	int width, height;
	if(type == BIG_ARMOR)
	{
		width = 92;
		height = 50;
	}
	else
	{
		width = 50;
		height = 50;
	}

	Point2f src[4]{Vec2f(tl), Vec2f(tr), Vec2f(br), Vec2f(bl)};
	Point2f dst[4]{Point2f(0.0, 0.0), Point2f(width, 0.0), Point2f(width, height), Point2f(0.0, height)};
	const Mat perspMat = getPerspectiveTransform(src, dst);
	cv::warpPerspective(grayImg, frontImg, perspMat, Size(width, height));
}


ArmorDetector::ArmorDetector()
{
    _flag = ARMOR_NO;
    _roi = Rect(cv::Point(0, 0), _srcImg.size());
    _isTracking = false;

#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
    _debugWindowName = "debug info";
#endif // DEBUG_DETECTION || SHOW_RESULT
}

ArmorDetector::ArmorDetector(const ArmorParam & armorParam)
{
	_param = armorParam;
	_flag = ARMOR_NO;
	_roi = Rect(cv::Point(0, 0), _srcImg.size());
	_isTracking = false;

#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
	_debugWindowName = "debug info";
#endif // DEBUG_DETECTION || SHOW_RESULT
}

void ArmorDetector::init(const ArmorParam &armorParam)
{
    _param = armorParam;
}

void ArmorDetector::loadImg(const cv::Mat & srcImg)
{
	_srcImg = srcImg;

#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
	_debugImg = srcImg.clone();
#endif // DEBUG_DETECTION || SHOW_RESULT

	Rect imgBound = Rect(cv::Point(0, 0), _srcImg.size());

    if(_flag == ARMOR_LOCAL && _trackCnt != _param.max_track_num)
    {
        cv::Rect bRect = boundingRect(_targetArmor.vertex) + _roi.tl();
        bRect = cvex::scaleRect(bRect, Vec2f(3,2));	//以中心为锚点放大2倍
        _roi = bRect & imgBound;
        _roiImg = _srcImg(_roi).clone();
    }
    else
    {
		_roi = imgBound;
		_roiImg = _srcImg.clone();
		_trackCnt = 0;
    }

#ifdef DEBUG_DETECTION
	rectangle(_debugImg, _roi, cvex::YELLOW);
#endif // DEBUG_DETECTION
}


int ArmorDetector::detect()
{
	/*
	*	Detect lights and build light bars' desciptors
	*/
	_armors.clear();
	std::vector<LightDescriptor> lightInfos;
	{
		/*
		*	pre-treatment
		*/
        split(_roiImg, channels);   // split color channels
        // pretreat and delete the color of friend armor
        if (_enemy_color == RED)
            _grayImg = channels.at(2) - channels.at(0);   // get red-blue image;
        else _grayImg = channels.at(0) - channels.at(2);    // get blue-red image;
        
		cv::Mat binBrightImg;

		cv::threshold(_grayImg, binBrightImg, _param.brightness_threshold, 255, cv::THRESH_BINARY);

        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        dilate(binBrightImg, binBrightImg, element);

    #ifdef DEBUG_PRETREATMENT
        imshow("brightness_binary", binBrightImg);
        waitKey();
    #endif // DEBUG_PRETREATMENT

		/*
		*	find and filter light bars  (find the contours)
		*/
		vector<vector<Point>> lightContours;
		cv::findContours(binBrightImg.clone(), lightContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		for(const auto& contour : lightContours)
		{
			float lightContourArea = contourArea(contour);
			if(contour.size() <= 5 ||
			   lightContourArea < _param.light_min_area) continue;

			RotatedRect lightRec = fitEllipse(contour);
			adjustRec(lightRec, ANGLE_TO_UP);

			//float solidity = lightContourArea / lightRec.size.area();
            		if(lightRec.size.width / lightRec.size.height > _param.light_max_ratio ||
			   lightContourArea / lightRec.size.area() < _param.light_contour_min_solidity) continue;

			//Mat temp;
			//cvex::showRectangle("light_right_position", _srcImg, temp, lightRec, cvex::GREEN,0, _roi.tl());

			lightRec.size.width *= _param.light_color_detect_extend_ratio;
			lightRec.size.height *= _param.light_color_detect_extend_ratio;
			Rect lightRect = lightRec.boundingRect();
			const Rect srcBound(Point(0, 0), _roiImg.size());
			lightRect &= srcBound;
            // save the light information, or add another debug treatment.
			lightInfos.push_back(LightDescriptor(lightRec));
		}

	#ifdef DEBUG_DETECTION
		vector<RotatedRect> lightsRecs;
		for(auto & light : lightInfos)
		{
			lightsRecs.emplace_back(light.rec());
		}
        cvex::showRectangles(_debugWindowName, _debugImg, _debugImg, lightsRecs, cvex::MAGENTA, 1, _roi.tl());
	#endif //DEBUG_DETECTION

		if(lightInfos.empty())
		{
			return _flag = ARMOR_NO;
		}
	}


	/*
	*	find and filter light bar pairs
	*/

	{
        // sort the lights according to x coordinates
		sort(lightInfos.begin(), lightInfos.end(), [](const LightDescriptor& ld1, const LightDescriptor& ld2)
		{
			return ld1.center.x < ld2.center.x;
		});
		// vector<int> minRightIndices(lightInfos.size(), -1);
		for(size_t i = 0; i < lightInfos.size(); i++)
		{
			for(size_t j = i + 1; (j < lightInfos.size()); j++)
			{
				const LightDescriptor& leftLight  = lightInfos[i];
				const LightDescriptor& rightLight = lightInfos[j];

		#ifdef DEBUG_DETECTION
				Mat pairImg = _debugImg.clone();
				vector<RotatedRect> curLightPair{leftLight.rec(), rightLight.rec()};
                		cvex::showRectangles("debug pairing",  pairImg, pairImg,curLightPair, cvex::CYAN, 1, _roi.tl());
		#endif // DEBUG_DETECTION

				/*
				*	Works for 2-3 meters situation
				*	morphologically similar: // parallel 
								 // similar height
				*/
				float angleDiff_ = abs(leftLight.angle - rightLight.angle);
				float LenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
				if(angleDiff_ > _param.light_max_angle_diff_ ||
				   LenDiff_ratio > _param.light_max_height_diff_ratio_)
				{
					continue;
				}

				/*
				*	proper location: // y value of light bar close enough 
				*			 // ratio of length and width is proper
				*/
				float dis = cvex::distance(leftLight.center, rightLight.center);
				float meanLen = (leftLight.length + rightLight.length) / 2;
				float yDiff = abs(leftLight.center.y - rightLight.center.y);
				float yDiff_ratio = yDiff / meanLen;
				float xDiff = abs(leftLight.center.x - rightLight.center.x);
				float xDiff_ratio = xDiff / meanLen;
				float ratio = dis / meanLen;
				if(yDiff_ratio > _param.light_max_y_diff_ratio_ ||
				   xDiff_ratio < _param.light_min_x_diff_ratio_ ||
				   ratio > _param.armor_max_aspect_ratio_ ||
				   ratio < _param.armor_min_aspect_ratio_)
				{
					continue;
				}

				// calculate pairs' info 
                // TODO: according to 2020 rules, no need to distinguish armortype
				int armorType = ratio > _param.armor_big_armor_ratio ? BIG_ARMOR : SMALL_ARMOR;
				// calculate the rotation score
				float ratiOff = (armorType == BIG_ARMOR) ? max(_param.armor_big_armor_ratio - ratio, float(0)) : max(_param.armor_small_armor_ratio - ratio, float(0));
				float yOff = yDiff / meanLen;
				float rotationScore = -(ratiOff * ratiOff + yOff * yOff);

				ArmorDescriptor armor(leftLight, rightLight, armorType, _grayImg, rotationScore, _param);
				_armors.emplace_back(armor);
				break;
			}
		}

	#ifdef  DEBUG_DETECTION
		vector<vector<Point>> armorVertexs;
		for(const auto & armor : _armors)
		{
			vector<Point> intVertex;
			for(const auto& point : armor.vertex)
			{
				intVertex.emplace_back(Point(point.x,point.y));
			}
			armorVertexs.emplace_back(intVertex);
		}
        cvex::showContours(_debugWindowName, _debugImg, _debugImg,armorVertexs,  cvex::WHITE, 1, _roi.tl());
	#endif //  DEBUG_DETECTION
		
		if(_armors.empty())
		{
			return _flag = ARMOR_NO;
		}
	}

#ifdef GET_ARMOR_PIC
	_allCnt++;
	int i = 0;
	for(const auto & armor : _armors)
    	{
        	Mat regulatedFrontImg = armor.frontImg;
        	if(armor.type == BIG_ARMOR)
        	{
       		     	regulatedFrontImg = regulatedFrontImg(Rect(21, 0, 50, 50));
        	}
        	imwrite("/home/nvidia/Documents/ArmorTrainingSample/" + to_string(_allCnt) + "_" + to_string(i) + ".bmp", regulatedFrontImg);
		i++;
	}
#endif // GET_ARMOR_PIC

	//delete the fake armors
	_armors.erase(remove_if(_armors.begin(), _armors.end(), [](ArmorDescriptor& i)
	{
		return !(i.isArmorPattern());   // ricky: svm or template match
	}), _armors.end());

	if(_armors.empty())
	{
		_targetArmor.clear();

		if(_flag == ARMOR_LOCAL)
		{
			//cout << "Tracking lost" << endl;
			return _flag = ARMOR_LOST;
		}
		else
		{
			//cout << "No armor pattern detected." << endl;
			return _flag = ARMOR_NO;
		}
	}

	//calculate the final score
    // TODO 关于多个目标选取的算法以及通信规则还需要敲定
	for(auto & armor : _armors)
	{
		armor.finalScore = armor.sizeScore + armor.distScore + armor.rotationScore;
	}

	//choose the one with highest score, store it on _targetArmor
	std::sort(_armors.begin(), _armors.end(), [](const ArmorDescriptor & a, const ArmorDescriptor & b)
	{
		return a.finalScore > b.finalScore;
	});
	_targetArmor = _armors[0];

	//update the flag status	
	_trackCnt++;

#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
	vector<Point> intVertex;
	for(const auto& point : _targetArmor.vertex)
	{
		Point fuckPoint = point;
		intVertex.emplace_back(fuckPoint);
	}
   	cvex::showContour(_debugWindowName, _debugImg, _debugImg, intVertex, cvex::GREEN, -1, _roi.tl());
#endif //DEBUG_DETECTION || SHOW_RESULT

	return _flag = ARMOR_LOCAL;
}

// bool ArmorDescriptor::isArmorPattern() const
// {
//    // cut the central part of the armor
//    Mat regulatedImg;
//    if(type == BIG_ARMOR)
//    {
//        regulatedImg = frontImg(Rect(21, 0, 50, 50));
//    }
//    else
//    {
//        regulatedImg = frontImg;
//    }

//    resize(regulatedImg,regulatedImg, Size(regulatedImg.size().width / 2, regulatedImg.size().height / 2));
//    // copy the data to make the matrix continuous
//    Mat temp;
//    regulatedImg.copyTo(temp);
//    Mat data = temp.reshape(1, 1);

//    data.convertTo(data, CV_32FC1);

//    Ptr<SVM> svm = StatModel::load<SVM>("/home/nvidia/Robomaster/Robomaster2018/Armor/SVM3.xml");


//    int result = (int)svm->predict(data);
//    if(result == 1) return true;
//    else return false;
	
	
	// to test the svm, uncomment the code block above
	// and comment the code below
// 	return true;
// }

//模板匹配 根据装甲板中心的图案判断是不是装甲板
bool ArmorDescriptor::isArmorPattern(std::vector<cv::Mat> &small,
                                     std::vector<cv::Mat> &big ,
                                     LastenemyType &lastEnemy)
{
    //若需要判断装甲中间数字
// #ifdef IS_ARMOR
    vector<pair<int,double>> score;
    map<int,double> mp;
    Mat regulatedImg=frontImg;
	
    for(int i = 0; i < 8; i++){
        //载入模板，模板是在初始化的时候载入ArmorDetector类，
        //因为ArmorDescriptor与其非同类需要间接导入
        // TODO 在初始化程序载入模板变量: ./Armor/Template/
        Mat tepl = small[i];
        Mat tepl1 = big[i];
        cv::Point matchLoc;     //模板匹配得到位置，这里没用
        double value;   //模板匹配得分
		//匹配小装甲
        value = TemplateMatch(regulatedImg, tepl, matchLoc, CV_TM_CCOEFF_NORMED);
        mp[i+1] = value;
        score.push_back(make_pair(i+1,value));
		//匹配大装甲 TODO: no need for big Armor
        value = TemplateMatch(regulatedImg, tepl1, matchLoc, CV_TM_CCOEFF_NORMED);
        mp[i+11] = value;
        score.push_back(make_pair(i+11,value));
    }
    //对该装甲与所有模板匹配后的得分进行排序
    sort(score.begin(),score.end(), [](const pair<int,double> &a, const pair<int,double> &b)
    {
        return a.second > b.second;
    });
    //装甲中心位置
    cv::Point2f c=(vertex[0]+vertex[1]+vertex[2]+vertex[3])/4;
	//装甲数字即为得分最高的那个
    int resultNum=score[0].first;
    //得分太低认为没识别到数字
    if(score[0].second<0.6)
    {
        if(//与上次识别到的装甲板位置差不多，且丢失次数不超过一定值
                std::abs(std::abs(lastEnemy.center.x)-std::abs(c.x))<10&&
                std::abs(std::abs(lastEnemy.center.y)-std::abs(c.y))<10&&
                lastEnemy.lostTimes<100
                )
        {//认为该装甲的数字与上次相同
            lastEnemy.lostTimes++;
            lastEnemy.center=c;
            enemy_num=lastEnemy.num;
            return true;
        }
        else
        {//认为不是装甲
            return false;
        }
    }
    //当装甲板识别为小装甲，而得到的号码为11、22……时，说明把大装甲识别为了小装甲。
    if(type==SMALL_ARMOR )
    {
        if(resultNum>10)
        {
            type=BIG_ARMOR;
        }
    }
    enemy_num=resultNum%10;
    lastEnemy.num=enemy_num;
    lastEnemy.center=c;
    lastEnemy.lostTimes=0;
    return true;
// #endif
    //若不需要判断匹配的装甲板中的数字则整个函数直接返回true
// #ifndef IS_ARMOR
    // return true;
// #endif
}

const std::vector<cv::Point2f> ArmorDetector::getArmorVertex() const
{
	vector<cv::Point2f> realVertex;
	for(int i = 0; i < 4; i++)
	{
		realVertex.emplace_back(Point2f(_targetArmor.vertex[i].x + _roi.tl().x, 
										_targetArmor.vertex[i].y + _roi.tl().y));
	}
	return realVertex;
}

int ArmorDetector::getArmorType() const
{
	return _targetArmor.type;
}

#if defined(DEBUG_DETECTION) || defined(SHOW_RESULT)
void ArmorDetector::showDebugImg() const
{
	imshow(_debugWindowName, _debugImg);
}
#endif // DEBUG_DETECTION || SHOW_RESULT
}
