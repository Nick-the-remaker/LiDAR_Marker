#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>
#include <array>
#include <iomanip>

#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <pcl/common/common.h>
#include <pcl/console/print.h>
#include <pcl/search/kdtree.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/pcl_exports.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/features/boundary.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <tf2_msgs/TFMessage.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using namespace std;
typedef pcl::PointXYZI PointT;

const int k = 30; // 邻居点个数

using namespace Eigen;

const int SMOOTH_WINDOW = 7; // 平滑窗口大小（奇数）
const float PEAK_VALLEY_RATIO = 0.02f;

struct PlaneParams
{
    float a, b, c, d;
    Vector3f center;
};

// 边界点提取函数
pcl::PointCloud<pcl::PointXYZ>::Ptr extractBoundaryCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
    double normal_radius = 0.1,
    double boundary_radius = 0.1,
    float angle_threshold = M_PI / 2)
{
    // 创建输出点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 1. 法线估计
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setInputCloud(input_cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(normal_radius);
    ne.compute(*normals);

    // 2. 边界检测
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> be;
    pcl::PointCloud<pcl::Boundary>::Ptr boundaries(new pcl::PointCloud<pcl::Boundary>);

    be.setInputCloud(input_cloud);
    be.setInputNormals(normals);
    be.setSearchMethod(tree);
    be.setRadiusSearch(boundary_radius);
    be.setAngleThreshold(angle_threshold);
    be.compute(*boundaries);

    // 3. 提取边界点
    for (size_t i = 0; i < input_cloud->size(); ++i)
    {
        if ((*boundaries)[i].boundary_point != 0)
        {
            boundary_cloud->push_back((*input_cloud)[i]);
        }
    }

    return boundary_cloud;
}

// 移动平均平滑
std::vector<float> movingAverage(const std::vector<float> &data, int window)
{
    std::vector<float> smoothed(data.size());
    int half = window / 2;

    for (int i = 0; i < data.size(); ++i)
    {
        float sum = 0;
        int count = 0;
        for (int j = -half; j <= half; ++j)
        {
            if (i + j >= 0 && i + j < data.size())
            {
                sum += data[i + j];
                count++;
            }
        }
        smoothed[i] = sum / count;
    }
    return smoothed;
}

// 改进的极值检测函数（使用相对阈值）
std::vector<int> findValleys(const std::vector<float> &data)
{
    std::vector<int> valleys;
    const int min_interval = 20;

    // 计算平均波动范围
    float avg_range = 0;
    for (int i = 1; i < data.size(); ++i)
    {
        avg_range += fabs(data[i] - data[i - 1]);
    }
    avg_range /= data.size() - 1;
    const float threshold = avg_range * PEAK_VALLEY_RATIO;

    for (int i = 1; i < data.size() - 1; ++i)
    {
        bool isValley = (data[i] < data[i - 1] - threshold) &&
                        (data[i] < data[i + 1] - threshold);

        // 排除连续下降的情况
        bool isContinuous = (data[i] - data[i - 1]) < -threshold &&
                            (data[i + 1] - data[i]) > threshold;

        if (isValley || isContinuous)
        {
            // 检查最小间隔
            if (!valleys.empty() && (i - valleys.back()) < min_interval)
            {
                // 保留更深的波谷
                if (data[i] < data[valleys.back()])
                {
                    valleys.pop_back();
                    valleys.push_back(i);
                }
            }
            else
            {
                valleys.push_back(i);
            }
        }
    }
    return valleys;
}

// // 改进的极值检测
// std::vector<int> findExtremums(const std::vector<float>& data, bool findValley) {
//     std::vector<int> extremums;
//     const int min_interval = 30; // 最小极值间隔

//     for (int i=1; i<data.size()-1; ++i) {
//         bool isExtremum = findValley ?
//             (data[i] < data[i-1]-PEAK_VALLEY_THRESHOLD && data[i] < data[i+1]-PEAK_VALLEY_THRESHOLD) :
//             (data[i] > data[i-1]+PEAK_VALLEY_THRESHOLD && data[i] > data[i+1]+PEAK_VALLEY_THRESHOLD);

//         if (isExtremum) {
//             if (!extremums.empty() && (i - extremums.back()) < min_interval) {
//                 if (findValley ? (data[i] < data[extremums.back()]) : (data[i] > data[extremums.back()])) {
//                     extremums.pop_back();
//                 } else {
//                     continue;
//                 }
//             }
//             extremums.push_back(i);
//         }
//     }
//     return extremums;
// }

void computeBasisVectors(const Vector3f &normal, Vector3f &u, Vector3f &v)
{
    Vector3f n = normal.normalized();
    if (n.x() == 0 && n.y() == 0)
    {
        u = Vector3f(1, 0, 0);
    }
    else
    {
        u = Vector3f(n.y(), -n.x(), 0).normalized();
    }
    v = n.cross(u).normalized();
}

bool analyzeSquareFit(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                      const PlaneParams &params, std::vector<Vector3f> &corners)
{
    // 计算基向量
    Vector3f n(params.a, params.b, params.c);
    Vector3f u, v;
    computeBasisVectors(n, u, v);

    // 原始数据采集（保持原有循环不变）
    std::vector<float> raw_lengths(360);
    // 遍历所有角度
    for (int theta_deg = 0; theta_deg < 360; ++theta_deg)
    {
        float theta = theta_deg * M_PI / 180.0f;
        Vector3f u_theta = u * cos(theta) + v * sin(theta);
        Vector3f v_theta = -u * sin(theta) + v * cos(theta);

        float min_u = FLT_MAX, max_u = -FLT_MAX;
        float min_v = FLT_MAX, max_v = -FLT_MAX;

        for (const auto &p : *cloud)
        {
            Vector3f vec = p.getVector3fMap() - params.center;
            float proj_u = vec.dot(u_theta);
            float proj_v = vec.dot(v_theta);

            if (proj_u < min_u)
                min_u = proj_u;
            if (proj_u > max_u)
                max_u = proj_u;
            if (proj_v < min_v)
                min_v = proj_v;
            if (proj_v > max_v)
                max_v = proj_v;
        }

        float side_length = std::max(max_u - min_u, max_v - min_v);
        raw_lengths[theta_deg] = side_length;
        // cout << side_length << ", ";
    }

    // 数据平滑处理
    auto smoothed_lengths = movingAverage(raw_lengths, SMOOTH_WINDOW);
    // cout <<endl << "smoothed_lengths" << endl;
    // // 输出smoothed_lengths
    // for (int i = 0; i < smoothed_lengths.size(); ++i)
    // {
    //     std::cout << smoothed_lengths[i] << "  ";
    // }

    // TODO findValleys这个函数有大问题
    auto valleys = findValleys(smoothed_lengths);
    for (int index : valleys)
    {
        cout << "Valley found at index: " << index << ", value: " << smoothed_lengths[index] << endl;
    }
    // // 输出波谷
    // std::cout << "Valleys found at angles: ";
    // for (int valley : valleys)
    // {
    //     std::cout << valley << " ";
    // }

    // 验证波谷模式（应间隔约90度）
    std::vector<int> valid_valleys;
    for (size_t i = 1; i < valleys.size(); ++i)
    {
        int diff = valleys[i] - valleys[i - 1];
        if (abs(diff - 90) < 15)
        { // 允许±15度误差
            valid_valleys.push_back(valleys[i - 1]);
            if (i == valleys.size() - 1)
                valid_valleys.push_back(valleys[i]);
        }
    }

    // 计算平均边长（取两个最小波谷的平均）
    float sum_length = 0;
    std::vector<float> valley_lengths;
    for (int v : valid_valleys)
    {
        valley_lengths.push_back(smoothed_lengths[v]);
    }
    std::sort(valley_lengths.begin(), valley_lengths.end());

    cout << "valley_lengths " << valley_lengths.size() << endl;

    if (valley_lengths.empty())
    {
        return false;
    }

    cout << "valley_lengths[0] " << valley_lengths[0] << " valley_lengths[1]" << valley_lengths[1] << endl;

    // if (valley_lengths[0] > 0.36 || valley_lengths[0] < 0.3 || valley_lengths[valley_lengths.size()-1] > 0.36 || valley_lengths[valley_lengths.size()-1] < 0.3 || fabs(valley_lengths[0] - valley_lengths[valley_lengths.size()-1]) > 0.04)
    // {
    //     return false;
    // }

    // if (valley_lengths[0] > 0.49 || valley_lengths[0] < 0.4 || valley_lengths[valley_lengths.size() - 1] > 0.46 || valley_lengths[valley_lengths.size() - 1] < 0.4 || fabs(valley_lengths[0] - valley_lengths[valley_lengths.size() - 1]) > 0.04)
    // {
    //     return false;
    // }

    float avg_length = (valley_lengths[0] + valley_lengths[1]) / 2.0f;

    cout << avg_length << endl;

    // 寻找最佳角度（最接近平均值的波谷）
    int best_theta = 0;
    float min_diff = FLT_MAX;
    float final_length = 0;
    for (int v : valid_valleys)
    {
        if (fabs(smoothed_lengths[v] - avg_length) < min_diff)
        {
            min_diff = fabs(smoothed_lengths[v] - avg_length);
            best_theta = v;
            final_length = smoothed_lengths[v];
        }
    }

    cout << "min_diff: " << min_diff << endl;

    cout << best_theta << endl;

    // 计算正方形角点
    float theta = best_theta * M_PI / 180.0f;
    Vector3f u_theta = u * cos(theta) + v * sin(theta);
    Vector3f v_theta = -u * sin(theta) + v * cos(theta);

    float half_len = final_length / 2.0f;
    corners[0] = params.center + u_theta * half_len + v_theta * half_len;
    corners[1] = params.center - u_theta * half_len + v_theta * half_len;
    corners[2] = params.center - u_theta * half_len - v_theta * half_len;
    corners[3] = params.center + u_theta * half_len - v_theta * half_len;

    if (corners.empty())
        return false;

    cout << "corners: " << endl;
    for (const auto &corner : corners)
    {
        std::cout << corner.transpose() << std::endl;
    }
    // 输出参数
    std::cout << "拟合正方形参数:\n";
    std::cout << "边长: " << final_length << "\n";
    std::cout << "方向角度: " << best_theta << " 度\n";
    std::cout << "中心点: (" << params.center.x() << ", "
              << params.center.y() << ", " << params.center.z() << ")\n";

    // // 可视化
    // pcl::visualization::PCLVisualizer viewer("Square Fit");
    // viewer.addPointCloud(cloud, "cloud");
    // viewer.setBackgroundColor(0, 0, 0);

    // for (int i = 0; i < 4; ++i)
    // {
    //     int j = (i + 1) % 4;
    //     pcl::PointXYZ p1(corners[i].x(), corners[i].y(), corners[i].z());
    //     pcl::PointXYZ p2(corners[j].x(), corners[j].y(), corners[j].z());
    //     viewer.addLine<pcl::PointXYZ>(p1, p2, 255, 0, 0, "line" + std::to_string(i));
    // }

    // while (!viewer.wasStopped())
    // {
    //     viewer.spinOnce();
    // }
    return true;
}

Eigen::Vector4f fitPlaneWithRANSAC(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01); // 距离阈值（根据点云尺度调整）
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0)
    {
        PCL_ERROR("无法拟合平面\n");
        return Eigen::Vector4f::Zero();
    }

    // // 强制法向量指向原点的那一侧
    // if (coefficients->values[3] < 0)
    // {
    //     coefficients->values[0] *= -1;
    //     coefficients->values[1] *= -1;
    //     coefficients->values[2] *= -1;
    //     coefficients->values[3] *= -1;
    // }

    // Eigen::Vector3f normal(coefficients->values[0],
    //                        coefficients->values[1],
    //                        coefficients->values[2]);
    // cout << "coe  " << coefficients->values[0] << " " << coefficients->values[1] << " " << coefficients->values[2] << " " << coefficients->values[3] << endl;
    // normal.normalize();
    Eigen::Vector4f coefficients_f(coefficients->values[0],
                                   coefficients->values[1],
                                   coefficients->values[2],
                                   coefficients->values[3]);
    return coefficients_f;
}

struct ThresholdNeighbors
{
    int k;                                 // 动态的邻居点个数（由外部传入）
    int thresh;                            // 阈值
    std::vector<int> neighbor_series;      // 大小由k决定的序列数组
    std::vector<float> neighbor_distances; // 大小由k决定的距离数组

    // 主构造函数
    explicit ThresholdNeighbors(int k_val, int t = 0)
        : k(k_val),
          thresh(t),
          neighbor_series(k_val, 0),   // 初始化k个元素，默认值0
          neighbor_distances(k_val, 0) // 初始化k个元素，默认值0
    {
    }
};

// bool find_split_threshold(int intensity, vector<int> &numbers)
// {
//     if (numbers.empty())
//     {
//         throw std::invalid_argument("The input number list is empty.");
//     }

//     // 复制数组并排序
//     std::sort(numbers.begin(), numbers.end());

//     // 自适应定义“较低的数”的范围：
//     // 1. 计算所有数的平均值
//     // double mean = std::accumulate(numbers.begin(), numbers.end(), 0.0) / numbers.size();

//     // 2. 找到比平均值小的数的最大值，作为“较低数”的分界点
//     // int threshold = numbers[0]; // 初始化为最小值
//     int count = 0;
//     for (int i = 0; i < 8; i++)
//     {
//         if (abs(numbers[i] - intensity) < 25)
//         {
//             count++;
//         }
//     }
//     // cout<<"count " << count << endl;
//     // if (count >= 6)

//     if (count >= 6)
//         return true;
//     else
//         return false;
// }

int find_split_threshold(vector<int> &numbers)
{
    if (numbers.empty())
    {
        throw std::invalid_argument("The input number list is empty.");
    }

    // 复制数组并排序
    std::sort(numbers.begin(), numbers.end());

    // 自适应定义“较低的数”的范围：
    // 1. 计算所有数的平均值
    double mean = std::accumulate(numbers.begin(), numbers.end(), 0.0) / numbers.size();

    // 2. 找到比平均值小的数的最大值，作为“较低数”的分界点
    int threshold = numbers[0]; // 初始化为最小值
    for (int num : numbers)
    {
        // cout<<" " << num ;
        if (num < mean)
        {
            threshold = num;
        }
        else
        {
            break; // 一旦超出平均值，停止循环
        }
    }
    // cout<<endl<<threshold<<endl;
    // return threshold;
    return 35;
}

void update_index(int idx,
                  std::vector<int> &remaining_indices,
                  std::unordered_map<int, size_t> &index_to_position)
{
    // 从 remaining_indices 中删除idx
    // cout << "idx " << idx << endl;
    // cout << "index_to_position.size() " << index_to_position.size() << endl;

    size_t pos = index_to_position[idx];
    // cout << "pos " << pos << endl;
    int last_val = remaining_indices.back();

    remaining_indices[pos] = last_val; // 用最后一个元素覆盖pos
    // cout << "remaining_indices[pos]" << remaining_indices[pos] << endl;
    index_to_position[last_val] = pos; // 更新last_val的位置
    // cout << "index_to_position[last_val]" << index_to_position[last_val] << endl;
    remaining_indices.pop_back(); // 删除末尾
    // cout << "remaining_indices.size() " << remaining_indices.size() << endl;
    index_to_position.erase(idx); // 移除idx的映射
                                  // cout<<"index_to_position.size() "<<index_to_position.size()<<endl;
}

void update_bounding_box_indices(pcl::PointXYZI &point, float indices[6])
{
    // cout << point << " former";
    // for (int i = 0; i < 6; i++)
    // {
    // 	cout << indices[i] << " ";
    // }
    // cout << endl;
    if (point.x < indices[0])
        indices[0] = point.x;
    if (point.y < indices[1])
        indices[1] = point.y;
    if (point.z < indices[2])
        indices[2] = point.z;
    if (point.x > indices[3])
        indices[3] = point.x;
    if (point.y > indices[4])
        indices[4] = point.y;
    if (point.z > indices[5])
        indices[5] = point.z;
    // cout << "after";
    // for (int i = 0; i < 6; i++)
    // {
    // 	cout << indices[i] << " ";
    // }
    // cout << endl;
}

int main(int argc, char **argv)
{

    // double vicon_x = 1.673751624943098;
    // double vicon_y = 0.43741966337323857;
    // double vicon_z = 0.62049286045725;

    // tf2::Quaternion imu_quat(
    //     -0.0017926737293357847, 0.0008273284555650879, -0.03297840248608363, 0.9994541144134595);

    // 2
    //  double vicon_x = 2.1542807532585972;
    //  double vicon_y = 0.41242374384103037;
    //  double vicon_z = 0.6214800753492665;

    // tf2::Quaternion imu_quat(
    //     -0.0005302959218508233,
    //     -0.0003894914727441116,
    //     -0.005669254856401248,
    //     0.9999837131833705);

    // 3
    //  double vicon_x = 2.7249639015192617;
    //  double vicon_y = 0.4857506250735433;
    //  double vicon_z = 0.6232584995589064;

    // tf2::Quaternion imu_quat(
    // -0.0002306894243320912,0.0029638016699309205,-0.0068527076300133796,0.999972101140921);

    // normal 1
    // double vicon_x = 1.9655485660810361;
    // double vicon_y = 0.4665126188363728;
    // double vicon_z = 0.5480476270171525;

    // tf2::Quaternion imu_quat(
    //     -0.0008903772606861914, 0.0002599346956052848, -0.03878606170426255, 0.999247107116033);

    // normal 2
    // double vicon_x = 2.5200060889235965;
    // double vicon_y = 0.4601750476236846;
    // double vicon_z = 0.5448846484993473;

    // tf2::Quaternion imu_quat(
    //     0.0005019557458569206, 0.001285009726520562, -0.033917594241678034, 0.9994236807236904);

    // normal 3
    // double vicon_x = 2.9725106146274367;
    // double vicon_y = 0.4980864551153538;
    // double vicon_z = 0.5423026822138092;

    // tf2::Quaternion imu_quat(
    //     -0.0003917849337282535, 0.004023325622657459, -0.02155789934442964, 0.9997594292285296);

    // normal 4
    double vicon_x = 3.529552902165125;
    double vicon_y = 0.502786085806584;
    double vicon_z = 0.5371017025810838;

    tf2::Quaternion imu_quat(
        -0.0007333492453226709, 0.0034029285480416433, -0.01155957896290444, 0.999927126549921);

    double roll, pitch, yaw; // 定义存储r\p\y的容器
    tf2::Matrix3x3 m(imu_quat);
    m.getRPY(roll, pitch, yaw); // 进行转换
    cout.precision(5);
    cout << fixed << setprecision(5) << endl;
    cout << "vicon x " << vicon_x << " " << vicon_y << " " << vicon_z << endl;

    cout << "vicon yaw: " << yaw * 180 / 3.1415926 << " pitch: " << pitch * 180 / 3.1415926 << " roll: " << roll * 180 / 3.1415926 << endl;

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());

    std::vector<ThresholdNeighbors> threshold_neighbors_vector;

    pcl::PointCloud<PointT>::Ptr save_cloud(new pcl::PointCloud<PointT>());

    // visualization::PCLVisualizer viewer("3D Viewer");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_color(new pcl::PointCloud<pcl::PointXYZRGB>());
    srand(time(nullptr));

    // vicon坐标系，前面是x轴，左边是y轴，上面是z轴
    // visualization
    pcl::visualization::PCLVisualizer viewer("RegionGrowing Viewer");

    // pcl::io::loadPCDFile("/home/tdt/0610/1.pcd", *cloud);
    pcl::io::loadPCDFile("/home/tdt/log_folder/pcd_log/field_test/qr1.pcd", *cloud);

    // pcl::io::loadPCDFile("/home/ub//20.pcd", *cloud);

    float search_radius = 0.06; // 搜索半径 0.04 到0.05
    float thresh_radius = 0.08; // 阈值计算半径，五厘米直径的圆
    // float search_radius = 0.06; // 搜索半径 0.04 到0.05
    // float thresh_radius = 0.08; // 阈值计算半径，五厘米直径的圆
    int points_num = cloud->points.size();

    auto start = chrono::high_resolution_clock::now();
    // 创建KD树 耗时30ms到40ms
    pcl::KdTreeFLANN<pcl::PointXYZI> tree;
    // pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree.setInputCloud(cloud);
    cout << "points_num " << points_num << endl;

    vector<int> k_curr_seed_index_test(k, 0); // 存储单个点的邻居索引。
    vector<float> k_curr_seed_dis_test(k, 0);

    auto test1 = chrono::high_resolution_clock::now();

    std::vector<int> remaining_indices(points_num);
    std::unordered_map<int, size_t> index_to_position(points_num);

    for (int i = 0; i < cloud->size(); i++)
    {
        // if(cloud->points[i].x < -0.14 && cloud->points[i].x > -0.6 && cloud->points[i].y < 1.95 && cloud->points[i].y > 1.6 && cloud->points[i].z < 0.8 && cloud->points[i].z > 0.35)
        // {
        //     save_cloud->push_back(cloud->points[i]);
        // }

        remaining_indices[i] = i;
        index_to_position[i] = i;

        ThresholdNeighbors point(k);

        tree.nearestKSearch(cloud->points[i], k, k_curr_seed_index_test, k_curr_seed_dis_test);

        std::vector<int> neibours_vector;

        for (int j = 0; j < k_curr_seed_index_test.size(); j++)
        {
            if (k_curr_seed_dis_test[j] < thresh_radius * thresh_radius)
                neibours_vector.push_back(cloud->points[k_curr_seed_index_test[j]].intensity);
        }

        // cout << cloud->points[i].intensity << endl;
        int curr_neibour_dyn_thresh = find_split_threshold(neibours_vector);

        // bool curr_neibour_dyn_thresh = find_split_threshold(cloud->points[i].intensity, neibours_vector);

        // 当前种子是在白色还是黑色区域内,1为黑色区域，2为白色区域
        int seed_thresh = cloud->points[i].intensity < curr_neibour_dyn_thresh ? 1 : 2;

        point.neighbor_series = k_curr_seed_index_test;
        point.neighbor_distances = k_curr_seed_dis_test;
        // point.thresh = curr_neibour_dyn_thresh;
        point.thresh = seed_thresh;

        threshold_neighbors_vector.push_back(point);
    }

    auto test2 = chrono::high_resolution_clock::now();
    // cout << "time1 " << chrono::duration_cast<chrono::milliseconds>(test2 - test1).count() << endl;

    int seed_orginal = 0;
    int counter_0 = 0;
    int segment_laber(0);
    vector<int> segmen_num;             // 存储每个分割的点数
    vector<int> point_laber;            // 存储点的标签
    point_laber.resize(points_num, -1); // 初始化存储点的标签为-1

    while (counter_0 < points_num)
    {
        queue<int> seed;
        seed.push(seed_orginal);
        point_laber[seed_orginal] = segment_laber;
        pcl::PointXYZI seed_point = cloud->points[seed_orginal];
        pcl::PointCloud<PointT>::Ptr tmp_segment_cloud(new pcl::PointCloud<PointT>());
        tmp_segment_cloud->push_back(seed_point);

        float bounding_box_indices[6] = {seed_point.x, seed_point.y, seed_point.z, seed_point.x, seed_point.y, seed_point.z}; // 顺序为x_min.y_min,z_min, x_max, y_max, z_max

        if (remaining_indices.size() == 1)
            break;
        update_index(seed_orginal, remaining_indices, index_to_position);

        int counter_1(1);
        int all_neibours_num(0); // 所有邻居点的数量
        int neighbors_bigger_than_seed(0);
        auto time1 = chrono::high_resolution_clock::now();
        vector<int> is_neibour_index(points_num, -1);

        // cout << "seg " << segment_laber << endl;

        // 一次运行几十到几百毫秒不等
        while (!seed.empty())
        {
            int curr_seed = seed.front(); // 取出队列的第一个元素
            seed.pop();

            // 当前种子是在白色还是黑色区域内,1为黑色区域，2为白色区域
            int seed_status = threshold_neighbors_vector[curr_seed].thresh;

            for (int i = 0; i < threshold_neighbors_vector[curr_seed].neighbor_series.size(); i++)
            {
                int neighbor_idx = threshold_neighbors_vector[curr_seed].neighbor_series[i];

                // bool curr_neibour_status = threshold_neighbors_vector[neighbor_idx].thresh;
                int curr_neibour_status = threshold_neighbors_vector[neighbor_idx].thresh;

                if (threshold_neighbors_vector[curr_seed].neighbor_distances[i] > search_radius * search_radius) // 如果当前点已经被标记
                {
                    // TODO: all_neibours_num这里要重写
                    // if (threshold_neighbors_vector[curr_seed].neighbor_distances[i] < search_radius * search_radius)
                    // {
                    //     if (curr_neibour_status > seed_status)
                    //     {
                    //         neighbors_bigger_than_seed++;
                    //     }
                    //     all_neibours_num++;
                    // }
                    // curr_nebor_index++;
                    continue;
                }
                if (point_laber[neighbor_idx] != -1)
                {
                    continue;
                }

                bool neibour_judge = seed_status == curr_neibour_status ? true : false;
                // bool neibour_judge = curr_neibour_status;
                // cout<<neibour_judge<<" ";

                if (neibour_judge)
                {
                    seed.push(neighbor_idx);
                    point_laber[neighbor_idx] = segment_laber; // 标记当前点
                    counter_1++;
                    tmp_segment_cloud->push_back(cloud->points[neighbor_idx]);

                    update_index(neighbor_idx, remaining_indices, index_to_position);
                    update_bounding_box_indices(cloud->points[neighbor_idx], bounding_box_indices);
                }
                else
                {
                    // cout << "enter" << endl;
                    if (is_neibour_index[neighbor_idx] == -1)
                    {
                        is_neibour_index[neighbor_idx] = 1;
                        // all_neibours_num++;
                        // // cout << "all_neibours_num " << all_neibours_num << endl;
                        // if (curr_neibour_status > seed_status)
                        // {
                        //     // cout << " neighbors_bigger_than_seed " << neighbors_bigger_than_seed << endl;
                        //     neighbors_bigger_than_seed++;
                        // }
                        // cout<<","<<neighbor_idx;
                    }
                }
            }
        }

        // pcl::io::savePCDFileASCII("/home/ub/log_folder/pcd_log/delete/" + std::to_string(segment_laber) + "_segment_cloud.pcd", *tmp_segment_cloud);

        float bounding_box_diagnol = sqrt(pow(bounding_box_indices[3] - bounding_box_indices[0], 2) + pow(bounding_box_indices[4] - bounding_box_indices[1], 2) + pow(bounding_box_indices[5] - bounding_box_indices[2], 2));

        int segment_size = tmp_segment_cloud->points.size();
        // cout << "segment_laber " << segment_laber << " segment_size " << segment_size << " bounding_box_diagnol " << bounding_box_diagnol << endl;

        // 对角线长度：正着的时候是0.4525，倾斜到最大角度是0.64

        // if (bounding_box_diagnol > 0.4 && bounding_box_diagnol < 0.7 && segment_size > 50 && segment_size < points_num / 4)
        // if (bounding_box_diagnol > 0.58 && bounding_box_diagnol < 0.8 && segment_size > 50 && segment_size < points_num / 4)
        if (segment_size > 50 && segment_size < points_num / 4)
        // if (segment_size > 50)
        {
            cout << "test " << segment_laber << " " << segment_size << endl;
            // if (segment_laber != 50)
            // {
            //     continue;
            // }

            pcl::PointXYZ center, normal_point, u_point, v_point;

            Eigen::Vector4f centroid;
            Eigen::Vector4f plane_coefficients;

            pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_cloud_XYZ(new pcl::PointCloud<pcl::PointXYZ>);

            // 转换为pcl::PointCloud<pcl::PointXYZ>
            for (size_t i = 0; i < tmp_segment_cloud->size(); ++i)
            {
                pcl::PointXYZ point;
                point.x = (*tmp_segment_cloud)[i].x;
                point.y = (*tmp_segment_cloud)[i].y;
                point.z = (*tmp_segment_cloud)[i].z;
                tmp_cloud_XYZ->push_back(point);
            }

            // cout << "save " << endl;
            // pcl::io::savePCDFileASCII("/home/ub/log_folder/pcd_log/delete/" + std::to_string(segment_laber) + "_seg.pcd", *tmp_cloud_XYZ);
            auto boundary_cloud = extractBoundaryCloud(tmp_cloud_XYZ);

            // 用边界点直接计算中心点，不用hull
            // 计算凸包
            // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
            // pcl::ConvexHull<pcl::PointXYZ> chull;
            // chull.setInputCloud(tmp_cloud_XYZ);
            // chull.reconstruct(*cloud_hull);

            plane_coefficients = fitPlaneWithRANSAC(tmp_segment_cloud);
            pcl::compute3DCentroid(*boundary_cloud, centroid);

            Eigen::Vector3f normal(plane_coefficients[0],
                                   plane_coefficients[1],
                                   plane_coefficients[2]);
            normal.normalize();

            PlaneParams params;

            params.a = plane_coefficients[0];
            params.b = plane_coefficients[1];
            params.c = plane_coefficients[2];
            params.d = plane_coefficients[3];
            params.center = Eigen::Vector3f(centroid[0], centroid[1], centroid[2]); // 中心点
            std::vector<Vector3f> corners(4);

            // if (params.center.x() < 3.5 && params.center.x() > 1.8 && params.center.z() > 0.2 && params.center.z() < 0.6 && analyzeSquareFit(boundary_cloud, params, corners))
            // if (params.center.x() < 0 && params.center.x() > -0.6 && params.center.z() > 0.2 && params.center.z() < 0.6 && analyzeSquareFit(boundary_cloud, params, corners))
            if (params.center.x() < 0.8 && params.center.x() > 0.5 && params.center.y() < -7 && params.center.z() > 0.2 && params.center.z() < 1.6 && analyzeSquareFit(boundary_cloud, params, corners))
            //  if(analyzeSquareFit(boundary_cloud, params, corners))
            {
                cout << "center " << params.center << endl;

                pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud(new pcl::PointCloud<pcl::PointXYZ>);

                corners[0].x() = 0.702173;
                corners[0].y() = -7.483246;
                corners[0].z() = 0.993967;

                corners[1].x() = 0.551471;
                corners[1].y() = -7.734267;
                corners[1].z() = 0.913404;

                corners[2].x() = 0.724091;
                corners[2].y() = -7.877593;
                corners[2].z() = 1.119120;

                corners[3].x() = 0.868667;
                corners[3].y() = -7.632705;
                corners[3].z() = 1.216360;


                // 矩形四个角点
                for (const auto &c : corners)
                {
                    box_cloud->push_back(pcl::PointXYZ(c.x(), c.y(), c.z()));
                }

                // box_cloud->push_back(pcl::PointXYZ(params.center.x(), params.center.y(), params.center.z()));

                box_cloud->push_back(pcl::PointXYZ(0.710420, -7.699174, 1.057414));

                // if (segment_laber == 50)
                // pcl::io::savePCDFileASCII("/home/ub/log_folder/pcd_log/delete/se_"+std::to_string(segment_laber)+".pcd", *tmp_segment_cloud);

                viewer.addLine<pcl::PointXYZ>(box_cloud->points[0], box_cloud->points[1], 255, 255, 255, "line1");
                viewer.addLine<pcl::PointXYZ>(box_cloud->points[1], box_cloud->points[2], 255, 255, 255, "line2");
                viewer.addLine<pcl::PointXYZ>(box_cloud->points[2], box_cloud->points[3], 255, 255, 255, "line3");
                viewer.addLine<pcl::PointXYZ>(box_cloud->points[3], box_cloud->points[0], 255, 255, 255, "line4");

                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "line1");
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "line2");
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "line3");
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "line4");

                // normal = -normal;

                // 打印结果
                std::cout << "=== Rectangle Fitting Result ===" << std::endl;

                // std::cout << "Length: " << length1 << " (ground truth: " << length << ")\n" << "Width: " << length2 << " (ground truth: " << width << ")\n";

                cout << "center: " << box_cloud->points[4] << endl;
                cout << "normal " << normal[0] << " " << normal[1] << " " << normal[2] << endl;

                // 添加线段表示法线

                Eigen::Vector3f corrected_u, corrected_v;

                Eigen::Vector3f p1 = corners[0];
                Eigen::Vector3f p2 = corners[1];
                Eigen::Vector3f p3 = corners[2];

                corrected_u = (p2 - p3).normalized();
                corrected_v = (p1 - p2).normalized();

                ////////////////////以下是分析部分/////////////////////

                cout << "corrected_u " << corrected_u[0] << " " << corrected_u[1] << " " << corrected_u[2] << endl;
                cout << "corrected_v " << corrected_v[0] << " " << corrected_v[1] << " " << corrected_v[2] << endl;

                // 创建绕Z轴旋转88.5度的初始旋转矩阵（右手法则，逆时针）
                // const float initial_yaw = -88.4f * M_PI / 180.0f; // 88.2
                const float initial_yaw = -89.5f * M_PI / 180.0f; // 88.2

                Eigen::Matrix3f R_z;
                R_z << cos(initial_yaw), -sin(initial_yaw), 0,
                    sin(initial_yaw), cos(initial_yaw), 0,
                    0, 0, 1;
                cout << "R_z " << R_z << endl;

                // R_z << -1.151, 0.053, 0,
                //     0.102, 0.998, 0,
                //     0, 0, 1;

                // Eigen::Vector3f R_z1( -1.151, 0.053, 0);
                // Eigen::Vector3f R_z2(0.102, 0.998, 0);
                // Eigen::Vector3f R_z3(0, 0, 1);

                // R_z.col(0) = R_z1.normalized();
                // R_z.col(1) = R_z2.normalized();
                // R_z.col(2) = R_z3.normalized();

                Eigen::Vector3f translation_vector(0, 0.1817, 0.0542);
                // Eigen::Vector3f translation_vector(0, 0.01617, 0.0542);

                Eigen::Vector3f center_vector(box_cloud->points[4].x, box_cloud->points[4].y, box_cloud->points[4].z);

                center_vector = R_z * center_vector + translation_vector;

                cout << "lidar: " << " x: " << center_vector[0] << " y: " << center_vector[1] << " z: " << center_vector[2] << endl;

                cout << "XYZ error: " << vicon_x - center_vector[0] << " " << vicon_y - center_vector[1] << " " << vicon_z - center_vector[2] << endl;

                // ================= 第二部分：构造新坐标系矩阵 =================
                // 定义变换后的坐标轴向量（列向量形式）
                Eigen::Vector3f new_x = normal;
                Eigen::Vector3f new_y = -corrected_v;
                Eigen::Vector3f new_z = corrected_u;

                // 构造新旋转矩阵并正交化
                Eigen::Matrix3f R_new;
                R_new.col(0) = new_x.normalized();
                R_new.col(1) = new_y.normalized();
                R_new.col(2) = new_z.normalized();

                // ================= 第三部分：计算相对旋转 =================
                // 计算相对旋转矩阵：R_relative = R_new * R_z^T
                Eigen::Matrix3f R_relative = R_new * R_z.transpose();

                // ================= 第四部分：计算欧拉角 =================
                // 使用Z-Y-X顺序分解相对旋转矩阵
                float yaw_for_analysis, pitch_for_analysis, roll_for_analysis;

                // 计算pitch（Y轴旋转）
                pitch_for_analysis = asin(-R_relative(2, 0));

                // 处理万向锁特殊情况
                if (abs(cos(pitch_for_analysis)) > 1e-7)
                {
                    // 计算yaw（Z轴旋转）
                    yaw_for_analysis = atan2(R_relative(1, 0), R_relative(0, 0));

                    // 计算roll（X轴旋转）
                    roll_for_analysis = atan2(R_relative(2, 1), R_relative(2, 2));
                }
                else
                {
                    yaw_for_analysis = 0.0;
                    roll_for_analysis = atan2(-R_relative(0, 1), R_relative(1, 1));
                }

                // ================= 第五部分：结果输出 =================
                const float rad_to_deg = 180.0f / M_PI;
                cout << "Lidar :" << " yaw: " << yaw_for_analysis * rad_to_deg << " pitch: " << pitch_for_analysis * rad_to_deg << " roll: " << roll_for_analysis * rad_to_deg << endl;
                cout << "yaw error: " << yaw_for_analysis * rad_to_deg - yaw * rad_to_deg - 180 << " pitch error: " << pitch_for_analysis * rad_to_deg - pitch * rad_to_deg << " roll error: " << roll_for_analysis * rad_to_deg - roll * rad_to_deg << endl;
                // float epsilon = 1e-6f;
                // if (std::abs(normal.dot(corrected_u)) < epsilon &&
                //     std::abs(normal.dot(corrected_v)) < epsilon &&
                //     std::abs(corrected_u.dot(corrected_v)) < epsilon)
                //     cout << "正交" << endl;

                // 法线的终点
                float scale = 0.5f; // 法线显示长度

                normal_point.x = box_cloud->points[4].x + normal[0] * scale;
                normal_point.y = box_cloud->points[4].y + normal[1] * scale;
                normal_point.z = box_cloud->points[4].z + normal[2] * scale;

                u_point.x = box_cloud->points[4].x + corrected_u[0] * scale;
                u_point.y = box_cloud->points[4].y + corrected_u[1] * scale;
                u_point.z = box_cloud->points[4].z + corrected_u[2] * scale;

                v_point.x = box_cloud->points[4].x + corrected_v[0] * scale;
                v_point.y = box_cloud->points[4].y + corrected_v[1] * scale;
                v_point.z = box_cloud->points[4].z + corrected_v[2] * scale;

                viewer.addLine<pcl::PointXYZ>(box_cloud->points[4], normal_point, 0.0, 0.0, 1.0, "normal");
                viewer.addLine<pcl::PointXYZ>(box_cloud->points[4], u_point, 1.0, 0.0, 0.0, "u");
                viewer.addLine<pcl::PointXYZ>(box_cloud->points[4], v_point, 0.0, 1.0, 0.0, "v");

                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "normal");
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "u");
                viewer.setShapeRenderingProperties
                (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "v");
            }
        }
        // cout << "seg " << segment_laber << " bounding_box_indices ";
        // for (int i = 0; i < 6; i++)
        // {
        // 	cout << bounding_box_indices[i] << " ";
        // }
        // cout << endl;
        unsigned char R = static_cast<unsigned char>(rand() % 256);
        unsigned char G = static_cast<unsigned char>(rand() % 256);
        unsigned char B = static_cast<unsigned char>(rand() % 256);
        int count = 0;
        for (int i = 0; i < cloud->points.size(); i++)
        {
            if (point_laber[i] == segment_laber)
            {
                count++;
                pcl::PointXYZRGB tmp;
                tmp.x = cloud->points[i].x;
                tmp.y = cloud->points[i].y;
                tmp.z = cloud->points[i].z;
                tmp.r = R;
                tmp.g = G;
                tmp.b = B;
                cloud_color->push_back(tmp);
            }
        }
        // cout<<"seg "<<segment_laber<<" count "<<count<<endl;
        auto time2 = chrono::high_resolution_clock::now();
        // cout << "time " << chrono::duration_cast<chrono::milliseconds>(time2 - time1).count() << endl;

        float ratio = 0.0;
        if (neighbors_bigger_than_seed > all_neibours_num || all_neibours_num > 1000000 || all_neibours_num == 0)
        {
            ratio = 0.0;
        }
        else
        {
            ratio = (float)neighbors_bigger_than_seed / (float)all_neibours_num;
        }
        // cout << "ratio " << ratio << " neighbors_bigger_than_seed " << neighbors_bigger_than_seed << " " << all_neibours_num << "  " << segment_laber << endl;

        segment_laber++;
        counter_0 += counter_1;
        segmen_num.push_back(counter_1);

        if (remaining_indices.size() > 0)
        {
            seed_orginal = remaining_indices[0];
        }
        // if(remaining_indices.size() == 1)
        // {
        // 	break;
        // }
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "total time " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;

    // pcl::io::savePCDFileASCII("/home/ub/log_folder/pcd_log/wer.pcd", *save_cloud);

    cout << "seg_num:" << segmen_num.size() << endl;
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_color(cloud, "intensity");
    viewer.addPointCloud(cloud_color);
    // viewer.addPointCloud(cloud, intensity_color, "original cloud");
    // viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original cloud");

    // pcl::PointXYZI a, b;
    // a.x = -0.14;
    // a.y = 2.98;
    // a.z = 0.946;

    // b.x = -0.14;
    // b.y = 3.155;
    // b.z = 0.946;

    // viewer->addLine<pcl::PointXYZI>(a, b, 255, 255, 255, "line1");
    // viewer->addArrow<pcl::PointXYZI>(a, b, 255, 0, 0, "arrow");
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    // viewer->spin();
    return 0;
}

