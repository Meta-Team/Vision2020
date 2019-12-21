#include <opencv2/core/core.hpp>    
#include <opencv2/highgui/highgui.hpp>    
using namespace cv;  
      
/* Confusing... */
/*
 * 用opencv读取USB摄像头数据并传入jetson
 * */

IplImage* read(int argc, char** argv) { 
    CvCapture* capture = cvCreateCameraCapture(1);
    IplImage* frame;

    frame = cvQueryFrame(capture);
    if(!frame) return NULL;

    char c = cvWaitKey(50);
    if(c==27) break;
 
    cvReleaseCapture(&capture);
    return frame;
}

// TOTO
void videoCap()
{
    VideoCapture usbCap(3);
    if (!usbCap.isOpened())
    {
        cout << "Cannot open the camera" << endl;
    }
    int i = 0;
    char imgName[100];
    namedWindow("usb video", WINDOW_NORMAL);
    bool start = false;
    bool stop = false;
    //Sleep(1000);
    while (!stop)
    {
        Mat frame;
        usbCap >> frame;
        imshow("usb video", frame);
        if ((waitKey(30) == 's') || start)
        {
            start = true;
            sprintf(imgName, "%s%d%s", "img", ++i, ".jpg");
            imwrite(imgName, frame);
            if (waitKey(30) == 'q')
                stop = true;
        }
    }
}

void videoRead()
{
    VideoCapture capture("1.avi");
    if (!capture.isOpened())
        cout << "error reading" << endl;
    char imgName [100];
    int i = 0;
    Mat frame;
    cvNamedWindow("video", CV_WINDOW_NORMAL);
    bool stop = false;
    while (!stop)
    {
        if (!capture.read(frame))
            break;
        imshow("video", frame);
        int key = waitKey(30);
        sprintf(imgName, "%s%d%s", "img", ++i, ".jpg");
        imwrite(imgName, frame);
    }   
}
