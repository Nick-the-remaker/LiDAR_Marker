#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace std;
using namespace cv;

static double angle(Point pt1, Point pt2, Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

int num = 2;

int main()
{

    cv::Mat objPoints(4, 1, CV_32FC3);
    float markerLength = 0.48;

    cv::Mat Camera_K = (cv::Mat_<double>(3, 3) << 912.1531982421875, 0.0, 652.1595458984375, 0.0, 911.8236083984375, 360.1410217285156, 0.0, 0.0, 1.0);
    cv::Mat Camera_D = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);

    int index = 0;

    while (index < 22)
    {
        Mat frame, frame_clone;
        index++;
        frame = imread("/home/tdt/0525/" + std::to_string(index) + ".jpg");

        Mat show_frame = frame.clone();
        cvtColor(frame, frame_clone, COLOR_BGR2GRAY);
        GaussianBlur(frame_clone, frame_clone, Size(3, 3), 0); // 进行高斯平滑

        Mat white = Mat::zeros(frame_clone.rows, frame_clone.cols, CV_8UC1);
        for (int i = 0; i < frame_clone.rows; i++)
        {
            // 获取第i行首地址,首地址又是第一个数据，是三通道，所以是Vec3b
            //  可以理解为三维数组，i个（j,3）维数组
            // Vec3b *src_rows_ptr = frame.ptr<Vec3b>(i);
            uchar *src_rows_ptr = frame_clone.ptr<uchar>(i);
            uchar *dst1_rows_ptr = white.ptr<uchar>(i);
            for (int j = 0; j < frame_clone.cols; j++)
            {

                int value = src_rows_ptr[j] * 3;
                if (value > 255)
                {
                    value = 255;
                }
                dst1_rows_ptr[j] = value;
            }
        }

        // int x = 120;
        // int y = 185;
        // cout << static_cast<int>(frame.at<Vec3b>(x, y)[0]) << " " << static_cast<int>(frame.at<Vec3b>(x, y)[1]) << " " << static_cast<int>(frame.at<Vec3b>(x, y)[2]) << endl;
        // cout << static_cast<int>(white.at<uchar>(x, y)) << endl;
        // circle(frame, cv::Point(x, y), 6, 255, 2);
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        // imshow("frame", white);

        Mat thre;
        threshold(frame, thre, 120, 255, THRESH_OTSU);
        // imshow("thresh", thre);
        // imshow("frame_clone", white);

        vector<vector<Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        findContours(thre, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        vector<vector<Point>> conPoly(contours.size());
        vector<Point2f> srcPts;

        std::vector<cv::Rect> all_rect;

        float y_sum = 0;
        for (int i = 0; i < contours.size(); i++)
        {

            // drawContours(show_frame, contours, i, Scalar(255), 2);
            double area = contourArea(contours[i]);
            // 面积小于100的不要，基本是噪点
            if (area > 2500 && area < 70000)
            {
                cv::Rect rect_roi = cv::boundingRect(contours[i]);
                if (fabs(rect_roi.width - rect_roi.height) > 200)
                    continue;
                double peri = arcLength(contours[i], true);
                // 使用多边形拟合函数approxPolyDP进行轮廓的四边形拟合
                approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
                srcPts = {conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3]};
                // drawContours(frame, contours, i, Scalar(255), 2);
                if (isContourConvex(Mat(srcPts))) // 凸性检测 检测一个曲线是不是凸的
                {
                    double maxCosine = 0;
                    for (int j = 2; j < 5; j++)
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(srcPts[j % 4], srcPts[j - 2], srcPts[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    if (maxCosine < 0.5)
                    {
                        bool judge = true;
                        float ratio = sqrt(pow(conPoly[i][0].x - conPoly[i][1].x, 2) + pow(conPoly[i][0].y - conPoly[i][1].y, 2)) / sqrt(pow(conPoly[i][2].x - conPoly[i][1].x, 2) + pow(conPoly[i][2].y - conPoly[i][1].y, 2));
                        if (ratio > 1)
                        {
                            ratio = 2 - ratio;
                        }
                        judge = ratio > 0.8;

                        if (judge && srcPts.size() == 4)
                        // if(true)
                        {

                            cout<<index<<endl;
                            cout << Point2f(srcPts[0]).x << ".000 " << Point2f(srcPts[0]).y <<".000"<< endl
                                 << Point2f(srcPts[1]).x << ".000 " << Point2f(srcPts[1]).y << ".000"<< endl
                                 << Point2f(srcPts[2]).x << ".000 " << Point2f(srcPts[2]).y <<".000"<< endl
                                 << Point2f(srcPts[3]).x << ".000 " << Point2f(srcPts[3]).y << ".000"<< endl;

                            cv::Mat rvecs, tvecs;

                            // TODO 注意！这里的矫正是不可取的，实际要读码才可以确定方向。

                            int width = rect_roi.width / 2 + rect_roi.tl().x;
                            int height = rect_roi.height / 2 + rect_roi.tl().y;
                            int T_L, T_R, B_R, B_L;

                            // 确定被矫正图形的左上，右上，左下，右下四个顶点，用于透视变换
                            for (int i = 0; i < srcPts.size(); i++)
                            {
                                if (srcPts[i].x < width && srcPts[i].y < height)
                                {
                                    T_L = i;
                                }
                                if (srcPts[i].x > width && srcPts[i].y < height)
                                {
                                    T_R = i;
                                }
                                if (srcPts[i].x > width && srcPts[i].y > height)
                                {
                                    B_R = i;
                                }
                                if (srcPts[i].x < width && srcPts[i].y > height)
                                {
                                    B_L = i;
                                }
                            }
                            // cout << T_L << " " << T_R << " " << B_R << " " << B_L << endl;
                            // cout << index << endl;
                            // cout << Point2f(srcPts[T_L]) << " " << Point2f(srcPts[T_R]) << " " << Point2f(srcPts[B_R]) << " " << Point2f(srcPts[B_L]) << endl;

                            // cout << Point2f(srcPts[T_L]).x << " " << Point2f(srcPts[T_L]).y << endl
                            //      << Point2f(srcPts[T_R]).x << " " << Point2f(srcPts[T_R]).y << endl
                            //      << Point2f(srcPts[B_R]).x << " " << Point2f(srcPts[B_R]).y <<endl
                            //      << Point2f(srcPts[B_L]).x << " " << Point2f(srcPts[B_L]).y << endl;

                            std::vector<cv::Point2f> AffinePts;
                            AffinePts.push_back(Point2f(srcPts[B_R]));

                            AffinePts.push_back(Point2f(srcPts[B_L]));
                            AffinePts.push_back(Point2f(srcPts[T_L]));
                            AffinePts.push_back(Point2f(srcPts[T_R]));

                            objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);

                            objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);
                            objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
                            objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);

                            solvePnP(objPoints, AffinePts, Camera_K, Camera_D, rvecs, tvecs, false, SOLVEPNP_IPPE);


                            for (int j = 0; j < 4; j++)
                            {
                                
                                line(show_frame, srcPts[j], srcPts[(j + 1) % 4], Scalar(0, 255, 0), 2, 8);
                                cv::circle(show_frame, srcPts[j], 5, Scalar(0, 0, 255), -1);
                                cv::drawFrameAxes(show_frame, Camera_K, Camera_D, rvecs, tvecs, markerLength * 1.2f, 2);
                            }
                        }
                    }
                }
            }
        }
        // video_ << frame;
        // imshow("image", frame);
        imshow("ssss", show_frame);
        // if (waitKey(0) == 's')
        // {
        cv::imwrite("/home/tdt/0509/" + std::to_string(num) + ".png", show_frame);
        // }
        waitKey(0);
    }
    return 0;
}