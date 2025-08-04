#include <ros/ros.h>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <unordered_set>

std::unordered_map<std::string, int> marker_codes;

static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

double EulerDistance(cv::Point pt1, cv::Point pt2)
{
    return sqrt((pt2.x - pt1.x) * (pt2.x - pt1.x) + (pt2.y - pt1.y) * (pt2.y - pt1.y));
}

cv::Point2f calculateExtendedPoint(const cv::Point2f &start,
                                   const cv::Point2f &end,
                                   float extensionFactor = 0.3125f)
{
    // extensionFactor: 1.3125f = 0.59388/0.45248
    cv::Point2f direction = end - start;

    float distance = std::sqrt(direction.x * direction.x + direction.y * direction.y);

    if (distance <= std::numeric_limits<float>::epsilon())
    {
        return end;
    }

    cv::Point2f unitVector = direction / distance;

    float extensionDistance = distance * extensionFactor;

    cv::Point2f extendedPoint = end + unitVector * extensionDistance;

    return extendedPoint;
}

// Calculate syndrome (s0, s1, s2)
std::tuple<bool, bool, bool> computeSyndrome(const std::string &code)
{
    // Extract grid data
    char A = code[0];
    char B = code[1];
    char C = code[2];
    char d3 = code[3]; // Second row, first data bit
    char d4 = code[4]; // Second row, second data bit
    char d5 = code[5]; // Second row, third data bit
    char d6 = code[6]; // Third row, first data bit
    char d7 = code[7]; // Third row, second data bit
    char d8 = code[8]; // Third row, third data bit

    // Calculate expected value for each parity bit
    char A_calc = ((d4 - '0') + (d5 - '0') + (d7 - '0') + (d8 - '0')) % 2 + '0';
    char B_calc = ((d3 - '0') + (d5 - '0') + (d6 - '0') + (d8 - '0')) % 2 + '0';
    char C_calc = ((d3 - '0') + (d4 - '0') + (d5 - '0')) % 2 + '0';

    // Calculate syndrome
    bool s0 = (A != A_calc);
    bool s1 = (B != B_calc);
    bool s2 = (C != C_calc);

    return std::make_tuple(s0, s1, s2);
}

// Hamming code check and correction function
bool hammingCheck(const std::string &input, std::string &output)
{
    // Check input length
    if (input.length() != 9)
    {
        std::cerr << "Error: Input string must be 9 characters long." << std::endl;
        output = input;
        return false;
    }

    // Calculate syndrome
    auto [s0, s1, s2] = computeSyndrome(input);

    // No error
    if (!s0 && !s1 && !s2)
    {
        output = input;
    }

    // Try to locate error position
    int errorPos = -1;

    // Determine error position based on syndrome
    if (s0 && s1 && s2)
    {
        errorPos = 5; // Second row, third data bit
    }
    else if (s0 && s1 && !s2)
    {
        errorPos = 8; // Third row, third data bit
    }
    else if (s0 && !s1 && s2)
    {
        errorPos = 4; // Second row, second data bit
    }
    else if (!s0 && s1 && s2)
    {
        errorPos = 3; // Second row, first data bit
    }
    else if (!s0 && !s1 && s2)
    {
        errorPos = 2; // Parity bit C
    }
    else if (s0 && !s1 && !s2)
    {
        // Could be parity bit A or data bit d7
        // Try flipping parity bit A
        std::string temp = input;
        temp[0] = (temp[0] == '0') ? '1' : '0';
        if (marker_codes.find(temp) != marker_codes.end())
        {
            output = temp;
        }
        else
        { // Otherwise flip data bit d7
            temp = input;
            temp[7] = (temp[7] == '0') ? '1' : '0';
            if (marker_codes.find(temp) != marker_codes.end())
            {
                output = temp;
            }
        }
    }
    else if (!s0 && s1 && !s2)
    {
        // Could be parity bit B or data bit d6
        // Try flipping parity bit B

        std::string temp = input;
        temp[1] = (temp[1] == '0') ? '1' : '0';
        if (marker_codes.find(temp) != marker_codes.end())
        {
            output = temp;
        }
        else
        { // Otherwise flip data bit d7
            temp = input;
            temp[6] = (temp[6] == '0') ? '1' : '0';
            if (marker_codes.find(temp) != marker_codes.end())
            {
                output = temp;
            }
        }
    }

    // If a definite error position is found, correct it
    if (errorPos != -1)
    {
        output = input;
        output[errorPos] = (output[errorPos] == '0') ? '1' : '0';
    }

    if (marker_codes.find(output) != marker_codes.end())
    {
        return true;
    }
    else
    {
        return false;
    }
}

// turn 3x3 grid into a string
std::string gridToString(const std::vector<std::vector<int>> &grid)
{
    std::string s;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            s += std::to_string(grid[i][j]);
        }
    }
    return s;
}

// rotate clockwise 90 degrees
std::vector<std::vector<int>> rotate90(const std::vector<std::vector<int>> &grid)
{
    std::vector<std::vector<int>> rotated(3, std::vector<int>(3));
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            rotated[i][j] = grid[2 - j][i];
        }
    }
    return rotated;
}

std::vector<cv::Point2f> getRotatedRectCorners(std::vector<cv::Point2f> &rect, int rotation)
{
    // Extract original corner points
    cv::Point2f tl = rect[0]; // Top-left corner
    cv::Point2f tr = rect[1]; // Top-right corner
    cv::Point2f br = rect[2]; // Bottom-right corner
    cv::Point2f bl = rect[3]; // Bottom-left corner

    // Return corners based on rotation orientation
    switch (rotation)
    {
    case 0: // Order: top-left, top-right, bottom-right, bottom-left
        return {tl, tr, br, bl};

    case 90: // 90° clockwise rotation
        // New top-left = original bottom-left
        // New top-right = original top-left
        // New bottom-right = original top-right
        // New bottom-left = original bottom-right
        return {bl, tl, tr, br};

    case 180:
        return {br, bl, tl, tr};

    case 270:
        return {tr, br, bl, tl};

    default:
        return {tl, tr, br, bl};
    }
}

void decode(cv::Mat &image, std::vector<cv::Point2f> &rect, std::vector<cv::Point2f> &new_corner, int &codeIndex)
{
    int gridWidth = image.cols / 3;
    int gridHeight = image.rows / 3;

    std::vector<std::vector<int>> grid(3, std::vector<int>(3, -1)); // 初始化3x3网格
    int index = 0;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            index++;
            // define the ROI for the current grid cell
            cv::Rect roi(j * gridWidth, i * gridHeight, gridWidth, gridHeight);
            cv::Mat cell = image(roi);
            // cv::imshow("cell" + to_string(i) + to_string(j), cell);

            // count the number of white pixels in the cell
            int whitePixels = countNonZero(cell);
            double ratio = static_cast<double>(whitePixels) / (gridWidth * gridHeight);

            // set grid value based on the ratio
            if (ratio > 0.5)
            {
                grid[i][j] = 1;
            }
            else if (ratio < 0.5)
            {
                grid[i][j] = 0;
            }
        }
    }

    std::string s0 = gridToString(grid);
    std::string s0_output, s90_output, s180_output, s270_output;
    hammingCheck(s0, s0_output);

    // rotate 90 degrees clockwise
    std::vector<std::vector<int>> grid90 = rotate90(grid);
    std::string s90 = gridToString(grid90);
    hammingCheck(s0, s90_output);

    /// 改 hammingcheck和搜索合在一起

    if (marker_codes.find(s0_output) != marker_codes.end())
    {
        // cout << "0 degree: " << s0 << endl;
        new_corner = getRotatedRectCorners(rect, 90);
        codeIndex = marker_codes[s0];
    }

    if (marker_codes.find(s90) != marker_codes.end())
    {
        // cout << "90 degree: " << s90 << endl;
        new_corner = getRotatedRectCorners(rect, 180);
        codeIndex = marker_codes[s90];
    }

    // rotate 180 degrees
    std::vector<std::vector<int>> grid180 = rotate90(grid90);
    std::string s180 = gridToString(grid180);
    if (marker_codes.find(s180) != marker_codes.end())
    {
        // cout << "180 degree: " << s180 << endl;
        new_corner = getRotatedRectCorners(rect, 270);
        codeIndex = marker_codes[s180];
    }

    // rotate 270 degrees
    std::vector<std::vector<int>> grid270 = rotate90(grid180);
    std::string s270 = gridToString(grid270);
    if (marker_codes.find(s270) != marker_codes.end())
    {
        // cout << "270 degree: " << s270 << endl;
        new_corner = getRotatedRectCorners(rect, 0);
        codeIndex = marker_codes[s270];
    }
}

void find_lidar_marker_image(cv::Mat &input_image, cv::InputArray cameraMatrix, cv::InputArray distCoeffs, cv::OutputArray rvec, cv::OutputArray tvec, float markerLength, bool draw_axis, int &code_index)
{
    cv::Mat objPoints(4, 1, CV_32FC3);
    cv::Mat image_clone, image_thre;

    cvtColor(input_image, image_clone, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(image_clone, image_clone, cv::Size(3, 3), 0); // 进行高斯平滑

    cv::threshold(image_clone, image_thre, 0, 255, cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(image_thre, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> conPoly(contours.size());
    std::vector<cv::Point2f> srcPts;

    std::vector<cv::Rect> all_rect;

    double image_area = input_image.cols * input_image.rows;

    for (int i = 0; i < contours.size(); i++)
    {
        // drawContours(show_frame, contours, i, Scalar(255), 2);
        double area = contourArea(contours[i]);
        // the size of the marker is about 0.42*0.42, the black border of the marker is 0.32*0.32, which is 58% of the marker, so the area should be less than 0.7*image_area
        if (area > 900 && area < 0.7 * image_area)
        {
            cv::Rect rect_roi = cv::boundingRect(contours[i]);
            if (fabs(rect_roi.width - rect_roi.height) > 200)
                continue;
            double peri = arcLength(contours[i], true);
            // approxPolyDP to fit the contour to a polygon
            approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
            srcPts = {conPoly[i][0], conPoly[i][1], conPoly[i][2], conPoly[i][3]};
            // drawContours(frame, contours, i, Scalar(255), 2);
            if (cv::isContourConvex(cv::Mat(srcPts)) && srcPts.size() == 4)
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
                    // if the ratio is too large, it is not a square
                    float ratio = sqrt(pow(conPoly[i][0].x - conPoly[i][1].x, 2) + pow(conPoly[i][0].y - conPoly[i][1].y, 2)) / sqrt(pow(conPoly[i][2].x - conPoly[i][1].x, 2) + pow(conPoly[i][2].y - conPoly[i][1].y, 2));
                    if (ratio > 1)
                    {
                        ratio = 2 - ratio;
                    }
                    if (ratio > 0.8 && srcPts.size() == 4)
                    {

                        int width = rect_roi.width / 2 + rect_roi.tl().x;
                        int height = rect_roi.height / 2 + rect_roi.tl().y;
                        int T_L = 0, T_R = 0, B_R = 0, B_L = 0;

                        cv::Point center_point((srcPts[0].x + srcPts[1].x + srcPts[2].x + srcPts[3].x) / 4, (srcPts[0].y + srcPts[1].y + srcPts[2].y + srcPts[3].y) / 4);

                        for (int k = 0; k < srcPts.size(); k++)
                        {
                            if (srcPts[k].x < width && srcPts[k].y < height)
                            {
                                T_L = k;
                            }
                            if (srcPts[k].x > width && srcPts[k].y < height)
                            {
                                T_R = k;
                            }
                            if (srcPts[k].x > width && srcPts[k].y > height)
                            {
                                B_R = k;
                            }
                            if (srcPts[k].x < width && srcPts[k].y > height)
                            {
                                B_L = k;
                            }
                        }

                        std::vector<cv::Point2f> black_corner_pts = {cv::Point2f(srcPts[T_L]), cv::Point2f(srcPts[T_R]), cv::Point2f(srcPts[B_R]), cv::Point2f(srcPts[B_L])};

                        srcPts[T_L] = calculateExtendedPoint(center_point, srcPts[T_L], 0.3125f);
                        srcPts[T_R] = calculateExtendedPoint(center_point, srcPts[T_R], 0.3125f);
                        srcPts[B_L] = calculateExtendedPoint(center_point, srcPts[B_L], 0.3125f);
                        srcPts[B_R] = calculateExtendedPoint(center_point, srcPts[B_R], 0.3125f);

                        // define the four corners of the marker
                        double LeftHeight = EulerDistance(srcPts[T_L], srcPts[B_L]);
                        double RightHeight = EulerDistance(srcPts[T_R], srcPts[B_R]);
                        double MaxHeight = std::max(LeftHeight, RightHeight);

                        double UpWidth = EulerDistance(srcPts[T_L], srcPts[T_R]);
                        double DownWidth = EulerDistance(srcPts[B_L], srcPts[B_R]);
                        double MaxWidth = std::max(UpWidth, DownWidth);

                        // calculate the perspective transform matrix
                        cv::Point2f SrcAffinePts[4] = {cv::Point2f(srcPts[T_L]), cv::Point2f(srcPts[T_R]), cv::Point2f(srcPts[B_R]), cv::Point2f(srcPts[B_L])};
                        cv::Point2f DstAffinePts[4] = {cv::Point2f(0, 0), cv::Point2f(MaxWidth, 0), cv::Point2f(MaxWidth, MaxHeight), cv::Point2f(0, MaxHeight)};
                        cv::Mat M = cv::getPerspectiveTransform(SrcAffinePts, DstAffinePts);

                        // perform the perspective transform
                        cv::Mat DstImg;
                        warpPerspective(image_thre, DstImg, M, cv::Point(MaxWidth, MaxHeight));

                        // verify the dstImg

                        cv::Mat white_border = DstImg(cv::Rect(0.119 * DstImg.cols, 0.119 * DstImg.rows, 0.7619 * DstImg.cols, 0.7619 * DstImg.rows)); // 0.119: 5/42, 0.7619: 32/42

                        cv::Mat black_border = DstImg(cv::Rect(0.238 * DstImg.cols, 0.238 * DstImg.rows, 0.5238 * DstImg.cols, 0.5238 * DstImg.rows)); // 0.238: 10/42, 0.5238: 22/42

                        int middle_count = countNonZero(white_border);

                        int white_border_count = cv::countNonZero(DstImg) - middle_count; // the nonzero pixels in the white border;

                        int black_border_count = middle_count - countNonZero(black_border); // the nonzero pixels in the black border;

                        float white_border_count_rate = float(white_border_count) / float(DstImg.cols * DstImg.rows - white_border.cols * white_border.rows);

                        float black_border_count_rate = float(black_border_count) / float(white_border.cols * white_border.rows - black_border.cols * black_border.rows);

                        // std::cout << "white_border_count: " << white_border_count << " black_border_count: " << black_border_count << " middle_count: " << middle_count << std::endl;

                        // std::cout << "white_border_count_rate " << white_border_count_rate << std::endl;

                        // std::cout << "black_border_count_rate " << black_border_count_rate << std::endl;

                        if (white_border_count_rate > 0.9 && black_border_count_rate < 0.1)
                        {
                            std::vector<cv::Point2f> rect_corners;
                            decode(black_border, black_corner_pts, rect_corners, code_index);

                            objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
                            objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
                            objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
                            objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);

                            solvePnP(objPoints, rect_corners, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);

                            if (draw_axis)
                            {
                                for (int j = 0; j < 4; j++)
                                {
                                    line(input_image, black_corner_pts[j], black_corner_pts[(j + 1) % 4], cv::Scalar(0, 255, 0), 2, 8);
                                    cv::circle(input_image, black_corner_pts[j], 5, cv::Scalar(0, 0, 255), -1);
                                    cv::drawFrameAxes(input_image, cameraMatrix, distCoeffs, rvec, tvec, markerLength * 1.2f, 2);
                                    cv::putText(input_image, "id = " + std::to_string(code_index), cv::Point(black_corner_pts[0].x, black_corner_pts[0].y - 20), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                                }
                            }

                            // cv::imshow("DstImg", DstImg);
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "matrix_reader_node");
    ros::NodeHandle nh;

    // get rotation_matrix
    std::vector<double> Camera_K;
    std::vector<double> Camera_D;
    std::string image_path;
    std::string video_path;
    bool use_image;

    std::vector<std::string> codes = {
        "100000010", "010000011", "010000100", "100000101", "111001000",
        "011001010", "101001011", "101001100", "001001110", "111001111",
        "101010000", "011010001", "111010011", "111010100", "001010101",
        "101010111", "010011000", "100011001", "110011010", "000011100",
        "110011101", "100011110", "010011111", "011100000", "101100001",
        "111100010", "001100011", "111100101", "011100111", "100101000",
        "010101001", "000101010", "110101100", "010101110", "100101111",
        "010110010", "100110011", "100110100", "010110101", "001111000",
        "111111001", "101111010", "011111011", "011111100", "111111110"};

    for (int i = 0; i < codes.size(); i++)
    {
        marker_codes[codes[i]] = i + 1;
    }

    if (!nh.getParam("Camera_K", Camera_K))
    {
        ROS_ERROR("Failed to get param 'Camera_K'");
        return -1;
    }

    if (!nh.getParam("Camera_D", Camera_D))
    {
        ROS_ERROR("Failed to get param 'Camera_D'");
        return -1;
    }

    if (!nh.getParam("input_image_path", image_path))
    {
        ROS_ERROR("Failed to get param 'input_image_path'");
        return -1;
    }

    if (!nh.getParam("input_video_path", video_path))
    {
        ROS_ERROR("Failed to get param 'input_video_path'");
        return -1;
    }

    if (!nh.getParam("use_image", use_image))
    {
        ROS_ERROR("Failed to get param 'use_image'");
        return -1;
    }

    // verify camera_k and camera_d
    if (Camera_K.size() != 9)
    {
        ROS_ERROR("Invalid Camera_K size. Expected 9 elements, got %zu", Camera_K.size());
        return -1;
    }

    if (Camera_D.size() != 5)
    {
        ROS_ERROR("Invalid Camera_D size. Expected 5 elements, got %zu", Camera_D.size());
        return -1;
    }

    // build opencv matrix
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << Camera_K[0], Camera_K[1], Camera_K[2],
                             Camera_K[3], Camera_K[4], Camera_K[5],
                             Camera_K[6], Camera_K[7], Camera_K[8]);

    cv::Mat camera_distcoeffs = (cv::Mat_<double>(1, 5) << Camera_D[0], Camera_D[1], Camera_D[2],
                                 Camera_D[3], Camera_D[4]);

    std::cout << "get param: " << std::endl;
    std::cout << "use_image: " << use_image << std::endl;
    std::cout << "image_path: " << image_path << std::endl;
    std::cout << "video_path: " << video_path << std::endl;
    std::cout << "camera_matrix: " << camera_matrix << std::endl;
    std::cout << "camera_distcoeffs: " << camera_distcoeffs << std::endl;

    // glob all the photos in the image_path
    if (use_image)
    {

        std::vector<cv::String> image_files;
        cv::glob(image_path, image_files, false);

        if (image_files.empty())
        {
            ROS_ERROR("No images found in this path: %s", image_path.c_str());
            return -1;
        }

        for (const auto &file : image_files)
        {
            cv::Mat input_image = cv::imread(file);
            if (input_image.empty())
            {
                ROS_ERROR("Failed to read image: %s", file.c_str());
                continue;
            }

            cv::Mat rvec, tvec;
            int code_id = -1;
            find_lidar_marker_image(input_image, camera_matrix, camera_distcoeffs, rvec, tvec, 0.48, true, code_id);
            std::cout << "code_id " << code_id << std::endl;
            cv::imshow("Input Image", input_image);
            cv::waitKey(0);
            // process rvec and tvec as needed
        }
    }
    else
    {
        // handle video processing
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened())
        {
            ROS_ERROR("Failed to open video file: %s", video_path.c_str());
            return -1;
        }
        else
        {

            cv::Mat frame;
            while (cap.read(frame))
            {
                if (!frame.empty())
                {
                    cv::Mat rvec, tvec;
                    int code_index = -1;
                    find_lidar_marker_image(frame, camera_matrix, camera_distcoeffs, rvec, tvec, 0.08, true, code_index);
                    // std::cout << "code_index: " << code_index << std::endl;
                    cv::imshow("Video Frame", frame);
                    cv::waitKey(20);
                }
                else
                {
                    ROS_WARN("Empty frame encountered in video stream.");
                }
            }
        }
    }

    ros::spinOnce();
    return 0;
}