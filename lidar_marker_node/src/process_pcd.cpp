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


#include <opencv2/opencv.hpp>

#include <tf2_msgs/TFMessage.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudNormalT;

const int k = 30;

struct PlaneParams
{
    float a, b, c, d;
    Vector3f center;
};

double distToSegment(double x, double y, double x1, double y1, double x2, double y2)
{
    double l2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    if (l2 < 1e-10)
    {
        return sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1));
    }
    double t = max(0.0, min(1.0, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2));
    double proj_x = x1 + t * (x2 - x1);
    double proj_y = y1 + t * (y2 - y1);
    return sqrt((x - proj_x) * (x - proj_x) + (y - proj_y) * (y - proj_y));
}

double distToSquare(double x, double y, double a)
{
    double dist_right = distToSegment(x, y, a, -a, a, a);    
    double dist_top = distToSegment(x, y, -a, a, a, a);      
    double dist_left = distToSegment(x, y, -a, a, -a, -a);   
    double dist_bottom = distToSegment(x, y, -a, -a, a, -a); 

    return min({dist_right, dist_top, dist_left, dist_bottom});
}

double computeLoss(const MatrixXd &points, double s, double theta)
{
    double a = s / 2.0; 
    double c = cos(theta);
    double s_theta = sin(theta);
    double loss = 0.0;

    for (int i = 0; i < points.rows(); i++)
    {
        double x = points(i, 0);
        double y = points(i, 1);

        double x_rot = x * c + y * s_theta;
        double y_rot = -x * s_theta + y * c;

        double d = distToSquare(x_rot, y_rot, a);
        loss += d * d; 
    }

    return loss;
}

Vector2d computeGradient(const MatrixXd &points, double s, double theta, double eps = 1e-4)
{
    double loss_plus_s = computeLoss(points, s + eps, theta);
    double loss_minus_s = computeLoss(points, s - eps, theta);
    double ds = (loss_plus_s - loss_minus_s) / (2 * eps);

    double loss_plus_theta = computeLoss(points, s, theta + eps);
    double loss_minus_theta = computeLoss(points, s, theta - eps);
    double dtheta = (loss_plus_theta - loss_minus_theta) / (2 * eps);

    return Vector2d(ds, dtheta);
}

Vector2d gradientDescent(const MatrixXd &points, double initial_s, double initial_theta,
                         double learning_rate = 0.01, int max_iter = 1000, double tol = 1e-6)
{
    double s = initial_s;
    double theta = initial_theta;
    double best_s = s;
    double best_theta = theta;
    double best_loss = computeLoss(points, s, theta);

    cout << "Initial loss: " << best_loss << endl;

    for (int iter = 0; iter < max_iter; iter++)
    {
        Vector2d grad = computeGradient(points, s, theta);

        s -= learning_rate * grad(0);
        theta -= learning_rate * grad(1);

        if (s < 0.01)
            s = 0.01;

        theta = fmod(theta, 2 * M_PI);
        if (theta < 0)
            theta += 2 * M_PI;

        double current_loss = computeLoss(points, s, theta);

        if (current_loss < best_loss)
        {
            best_loss = current_loss;
            best_s = s;
            best_theta = theta;
        }

        if (iter % 100 == 0)
        {
            cout << "Iter " << iter << ": s=" << s << ", theta=" << theta
                 << ", loss=" << current_loss << endl;
        }

        if (iter % 200 == 0)
        {
            learning_rate *= 0.8; 
        }
    }

    cout << "Optimization finished. Best loss: " << best_loss << endl;
    return Vector2d(best_s, best_theta);
}

void applyRandomTransform(PointCloudT::Ptr cloud)
{
    Matrix4f transform = Matrix4f::Identity();

    float angle_z = (M_PI / 4.0) * (2.0 * rand() / RAND_MAX - 1.0);
    Matrix4f rot_z = Matrix4f::Identity();
    rot_z(0, 0) = cos(angle_z);
    rot_z(0, 1) = -sin(angle_z);
    rot_z(1, 0) = sin(angle_z);
    rot_z(1, 1) = cos(angle_z);

    transform = rot_z;

    transform(0, 3) = 2.0 * (2.0 * rand() / RAND_MAX - 1.0);
    transform(1, 3) = 2.0 * (2.0 * rand() / RAND_MAX - 1.0);
    transform(2, 3) = 1.0 * (2.0 * rand() / RAND_MAX - 1.0);

    pcl::transformPointCloud(*cloud, *cloud, transform);
}

Vector3f computeCentroid(const PointCloudT::Ptr &cloud)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    return Vector3f(centroid[0], centroid[1], centroid[2]);
}

Vector3f computeNormalWithRANSAC(const PointCloudT::Ptr &cloud)
{
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    Vector3f normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    normal.normalize();

    Vector3f centroid = computeCentroid(cloud);
    Vector3f point_to_centroid(centroid.x() - cloud->points[0].x,
                               centroid.y() - cloud->points[0].y,
                               centroid.z() - cloud->points[0].z);
    if (normal.dot(point_to_centroid) < 0)
    {
        normal = -normal;
    }

    return normal;
}

void createLocalCoordinateSystem(const Vector3f &normal, Vector3f &u_axis, Vector3f &v_axis)
{
    Vector3f n = normal.normalized();

    Vector3f basis(1, 0, 0);
    if (abs(n.dot(basis)) > 0.9)
        basis = Vector3f(0, 1, 0);

    u_axis = n.cross(basis).normalized();
    v_axis = n.cross(u_axis).normalized();

    if (abs(u_axis.dot(v_axis)) > 1e-3 || abs(u_axis.dot(n)) > 1e-3 || abs(v_axis.dot(n)) > 1e-3)
    {
        cout << "Warning: Coordinate system not orthogonal!" << endl;
        cout << "u·v: " << u_axis.dot(v_axis) << ", u·n: " << u_axis.dot(n) << ", v·n: " << v_axis.dot(n) << endl;
    }
}

MatrixXd projectToPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Vector3f &centroid, const Vector3f &normal, const Vector3f &u_axis, const Vector3f &v_axis)
{
    MatrixXd points(cloud->size(), 2);
    for (size_t i = 0; i < cloud->size(); i++)
    {
        Vector3f p(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
        Vector3f vec = p - centroid;

        Vector3f proj = vec - vec.dot(normal) * normal;

        points(i, 0) = proj.dot(u_axis);
        points(i, 1) = proj.dot(v_axis);
    }

    return points;
}

vector<Vector3f> computeSquareVertices(double s, double theta, const Vector3f &centroid,
                                       const Vector3f &normal, const Vector3f &u_axis, const Vector3f &v_axis)
{
    double a = s / 2.0; 
    double c = cos(theta);
    double s_theta = sin(theta);

    vector<Vector2d> local_vertices = {
        {a, a},   
        {-a, a},  
        {-a, -a}, 
        {a, -a}   
    };

    for (auto &v : local_vertices)
    {
        double x = v[0];
        double y = v[1];
        v[0] = x * c - y * s_theta; 
        v[1] = x * s_theta + y * c;
    }

    vector<Vector3f> vertices;
    for (const auto &v : local_vertices)
    {
        Vector3f p3d = centroid + v[0] * u_axis + v[1] * v_axis;

        Vector3f vec = p3d - centroid;
        double dist_to_plane = abs(vec.dot(normal));
        if (dist_to_plane > 1e-5)
        {
            cout << "Warning: Vertex not on plane! Distance: " << dist_to_plane << endl;
            p3d = p3d - dist_to_plane * normal;
        }

        vertices.push_back(p3d);
    }

    return vertices;
}

void compute2DBoundingBox(const MatrixXd &points, double &min_x, double &max_x, double &min_y, double &max_y)
{
    min_x = points(0, 0);
    max_x = points(0, 0);
    min_y = points(0, 1);
    max_y = points(0, 1);

    for (int i = 1; i < points.rows(); i++)
    {
        if (points(i, 0) < min_x)
            min_x = points(i, 0);
        if (points(i, 0) > max_x)
            max_x = points(i, 0);
        if (points(i, 1) < min_y)
            min_y = points(i, 1);
        if (points(i, 1) > max_y)
            max_y = points(i, 1);
    }
}

Eigen::Vector4f fitPlaneWithRANSAC(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0)
    {
        PCL_ERROR("Error fitting plane\n");
        return Eigen::Vector4f::Zero();
    }

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
    int k;                              
    int thresh;                           
    std::vector<int> neighbor_series;      
    std::vector<float> neighbor_distances; 

    explicit ThresholdNeighbors(int k_val, int t = 0)
        : k(k_val),
          thresh(t),
          neighbor_series(k_val, 0),  
          neighbor_distances(k_val, 0) 
    {
    }
};

int find_split_threshold(vector<int> &numbers)
{
    if (numbers.empty())
    {
        throw std::invalid_argument("The input number list is empty.");
    }


    std::sort(numbers.begin(), numbers.end());

    double mean = std::accumulate(numbers.begin(), numbers.end(), 0.0) / numbers.size();

    int threshold = numbers[0]; 
    for (int num : numbers)
    {
        if (num < mean)
        {
            threshold = num;
        }
        else
        {
            break; 
        }
    }
    // cout<<endl<<threshold<<endl;
    return threshold;
}

void update_index(int idx,
                  std::vector<int> &remaining_indices,
                  std::unordered_map<int, size_t> &index_to_position)
{

    size_t pos = index_to_position[idx];
    // cout << "pos " << pos << endl;
    int last_val = remaining_indices.back();

    remaining_indices[pos] = last_val;
    // cout << "remaining_indices[pos]" << remaining_indices[pos] << endl;
    index_to_position[last_val] = pos; 
    // cout << "index_to_position[last_val]" << index_to_position[last_val] << endl;
    remaining_indices.pop_back(); 
    // cout << "remaining_indices.size() " << remaining_indices.size() << endl;
    index_to_position.erase(idx); 
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

pcl::PointCloud<pcl::PointXYZ>::Ptr extractBoundaryCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
    double normal_radius = 0.1,
    double boundary_radius = 0.1,
    float angle_threshold = M_PI / 2)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setInputCloud(input_cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(normal_radius);
    ne.compute(*normals);

    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> be;
    pcl::PointCloud<pcl::Boundary>::Ptr boundaries(new pcl::PointCloud<pcl::Boundary>);

    be.setInputCloud(input_cloud);
    be.setInputNormals(normals);
    be.setSearchMethod(tree);
    be.setRadiusSearch(boundary_radius);
    be.setAngleThreshold(angle_threshold);
    be.compute(*boundaries);

    for (size_t i = 0; i < input_cloud->size(); ++i)
    {
        if ((*boundaries)[i].boundary_point != 0)
        {
            boundary_cloud->push_back((*input_cloud)[i]);
        }
    }

    return boundary_cloud;
}

bool analyzeSquareFit(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                      const PlaneParams &params, std::vector<Vector3f> &corners)
{
    Vector3f centroid = params.center;
    cout << "Centroid: (" << centroid[0] << ", " << centroid[1] << ", " << centroid[2] << ")" << endl;

    Vector3f normal(params.a, params.b, params.c);
    cout << "Estimated normal: (" << normal[0] << ", " << normal[1] << ", " << normal[2] << ")" << endl;

    Vector3f u_axis, v_axis;
    createLocalCoordinateSystem(normal, u_axis, v_axis);
    cout << "u_axis: (" << u_axis[0] << ", " << u_axis[1] << ", " << u_axis[2] << ")" << endl;
    cout << "v_axis: (" << v_axis[0] << ", " << v_axis[1] << ", " << v_axis[2] << ")" << endl;

    MatrixXd points_2d = projectToPlane(cloud, centroid, normal, u_axis, v_axis);

    double min_x, max_x, min_y, max_y;
    compute2DBoundingBox(points_2d, min_x, max_x, min_y, max_y);
    double width = max_x - min_x;
    double height = max_y - min_y;
    double initial_size = max(width, height) * 0.9; 
    double initial_theta = 0.0;       

    cout << "2D Bounding Box: width=" << width << ", height=" << height << endl;
    cout << "Starting optimization with initial size: " << initial_size << endl;

    Vector2d param = gradientDescent(points_2d, initial_size, initial_theta, 0.01, 1000);

    double fitted_size = param[0];
    double fitted_theta = param[1];

    cout << "\nFitting results:" << endl;
    cout << "  Fitted size: " << fitted_size << endl;
    cout << "  Fitted rotation angle: " << fitted_theta << " radians ("
         << fitted_theta * 180.0 / M_PI << " degrees)" << endl;

    corners = computeSquareVertices(fitted_size, fitted_theta,
                                    centroid, normal, u_axis, v_axis);

    if (corners.empty())
    {
        return false;
    }
    else
    {
        return true;
    }
}

int main()
{
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());

    std::vector<ThresholdNeighbors> threshold_neighbors_vector;

    pcl::PointCloud<PointT>::Ptr save_cloud(new pcl::PointCloud<PointT>());

    // visualization::PCLVisualizer viewer("3D Viewer");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_color(new pcl::PointCloud<pcl::PointXYZRGB>());
    srand(time(nullptr));

    // visualization
    pcl::visualization::PCLVisualizer viewer("RegionGrowing Viewer");

    pcl::io::loadPCDFile("/home/ub/log_folder/pcd_log/vicon_normal1.pcd", *cloud);

    float search_radius = 0.06; // 0.04 0.05
    float thresh_radius = 0.08; 

    int points_num = cloud->points.size();

    auto start = chrono::high_resolution_clock::now();
    pcl::KdTreeFLANN<pcl::PointXYZI> tree;
    // pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree.setInputCloud(cloud);
    cout << "points_num " << points_num << endl;

    vector<int> k_curr_seed_index_test(k, 0);
    vector<float> k_curr_seed_dis_test(k, 0);

    auto test1 = chrono::high_resolution_clock::now();

    std::vector<int> remaining_indices(points_num);
    std::unordered_map<int, size_t> index_to_position(points_num);

    for (int i = 0; i < cloud->size(); i++)
    {
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
    vector<int> segmen_num;          
    vector<int> point_laber;          
    point_laber.resize(points_num, -1); 

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
        int all_neibours_num(0);
        int neighbors_bigger_than_seed(0);
        auto time1 = chrono::high_resolution_clock::now();
        vector<int> is_neibour_index(points_num, -1);

        // cout << "seg " << segment_laber << endl;

        while (!seed.empty())
        {
            int curr_seed = seed.front(); 
            seed.pop();

            int seed_status = threshold_neighbors_vector[curr_seed].thresh;

            for (int i = 0; i < threshold_neighbors_vector[curr_seed].neighbor_series.size(); i++)
            {
                int neighbor_idx = threshold_neighbors_vector[curr_seed].neighbor_series[i];

                // bool curr_neibour_status = threshold_neighbors_vector[neighbor_idx].thresh;
                int curr_neibour_status = threshold_neighbors_vector[neighbor_idx].thresh;

                if (threshold_neighbors_vector[curr_seed].neighbor_distances[i] > search_radius * search_radius) // 如果当前点已经被标记
                {
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

        float bounding_box_diagnol = sqrt(pow(bounding_box_indices[3] - bounding_box_indices[0], 2) + pow(bounding_box_indices[4] - bounding_box_indices[1], 2) + pow(bounding_box_indices[5] - bounding_box_indices[2], 2));

        int segment_size = tmp_segment_cloud->points.size();

        // if (bounding_box_diagnol > 0.4 && bounding_box_diagnol < 0.7 && segment_size > 50 && segment_size < points_num / 4)
        // if (bounding_box_diagnol > 0.58 && bounding_box_diagnol < 0.8 && segment_size > 50 && segment_size < points_num / 4)
        if (segment_size > 50 && segment_size < points_num / 4)
        // if (segment_size > 50)
        {
            cout << "test " << segment_laber << " " << segment_size << endl;

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

            auto boundary_cloud = extractBoundaryCloud(tmp_cloud_XYZ);

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
            if (params.center.x() < 0 && params.center.x() > -0.4 && params.center.y() < 3 && params.center.z() > 0.2 && params.center.z() < 1 && analyzeSquareFit(boundary_cloud, params, corners))
            //  if(analyzeSquareFit(boundary_cloud, params, corners))
            {
                cout << "center " << params.center << endl;

                pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud(new pcl::PointCloud<pcl::PointXYZ>);

                for (const auto &c : corners)
                {
                    box_cloud->push_back(pcl::PointXYZ(c.x(), c.y(), c.z()));
                }

                box_cloud->push_back(pcl::PointXYZ(params.center.x(), params.center.y(), params.center.z()));

                viewer.addLine<pcl::PointXYZ>(box_cloud->points[0], box_cloud->points[1], 255, 255, 255, "line1");
                viewer.addLine<pcl::PointXYZ>(box_cloud->points[1], box_cloud->points[2], 255, 255, 255, "line2");
                viewer.addLine<pcl::PointXYZ>(box_cloud->points[2], box_cloud->points[3], 255, 255, 255, "line3");
                viewer.addLine<pcl::PointXYZ>(box_cloud->points[3], box_cloud->points[0], 255, 255, 255, "line4");

                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "line1");
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "line2");
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "line3");
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "line4");

                // normal = -normal;

                std::cout << "=== Rectangle Fitting Result ===" << std::endl;

                // std::cout << "Length: " << length1 << " (ground truth: " << length << ")\n" << "Width: " << length2 << " (ground truth: " << width << ")\n";

                cout << "center: " << box_cloud->points[4] << endl;
                cout << "normal " << normal[0] << " " << normal[1] << " " << normal[2] << endl;

                Eigen::Vector3f corrected_u, corrected_v;

                Eigen::Vector3f p1 = corners[0];
                Eigen::Vector3f p2 = corners[1];
                Eigen::Vector3f p3 = corners[2];

                corrected_u = (p2 - p3).normalized();
                corrected_v = (p1 - p2).normalized();

                float scale = 0.5f; 

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
                viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "v");
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
    // viewer.addPointCloud(cloud_color);
    viewer.addPointCloud(cloud, intensity_color, "original cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original cloud");

    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
    }

    return 0;
}