#ifndef SEGMATCH_ROS_COMMON_HPP_
#define SEGMATCH_ROS_COMMON_HPP_

#include <interactive_markers/interactive_marker_server.h>
#include <laser_slam/common.hpp>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <segmatch/common.hpp>
#include <segmatch/parameters.hpp>
#include <segmatch/segmatch.hpp>
#include <segmatch/segmented_cloud.hpp>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace segmatch_ros {

struct SegMatchWorkerParams {
  bool localize;
  bool close_loops;
  std::string target_cloud_filename;
  std::string world_frame;

  double distance_between_segmentations_m;
  double distance_to_lower_target_cloud_for_viz_m;

  bool align_target_map_on_first_loop_closure = false;
  segmatch::SegMatchParams segmatch_params;

  double ratio_of_points_to_keep_when_publishing;

  bool export_segments_and_matches = false;
}; // struct SegMatchWorkerParams


struct Color {
  Color(float red, float green, float blue) : r(red), g(green), b(blue) {}
  float r;
  float g;
  float b;
};

static void publishLineSet(const segmatch::PointPairs& point_pairs,
                           const std::string& frame, const float line_scale,
                           const Color& color, const ros::Publisher& publisher) {
  visualization_msgs::Marker line_list;
  line_list.header.frame_id = frame;
  line_list.header.stamp = ros::Time();
  line_list.ns = "matcher_trainer";
  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.action = visualization_msgs::Marker::ADD;
  line_list.color.r = color.r;
  line_list.color.g = color.g;
  line_list.color.b = color.b;
  line_list.color.a = 1.0;
  line_list.scale.x = line_scale;
  for (size_t i = 0u; i < point_pairs.size(); ++i) {
    geometry_msgs::Point p;
    p.x = point_pairs[i].first.x;
    p.y = point_pairs[i].first.y;
    p.z = point_pairs[i].first.z;
    line_list.points.push_back(p);
    p.x = point_pairs[i].second.x;
    p.y = point_pairs[i].second.y;
    p.z = point_pairs[i].second.z;
    line_list.points.push_back(p);
  }
  publisher.publish(line_list);
}

static void convert_to_pcl_point_cloud(const sensor_msgs::PointCloud2& cloud_message,
                                       segmatch::PointICloud* converted) {
  pcl::PCLPointCloud2 pcl_point_cloud_2;
  pcl_conversions::toPCL(cloud_message, pcl_point_cloud_2);
  pcl::fromPCLPointCloud2(pcl_point_cloud_2, *converted);
}

static void convert_to_point_cloud_2_msg(const segmatch::PointICloud& cloud,
                                         const std::string& frame,
                                         sensor_msgs::PointCloud2* converted) {
  CHECK_NOTNULL(converted);
  // Convert to PCLPointCloud2.
  pcl::PCLPointCloud2 pcl_point_cloud_2;
  pcl::toPCLPointCloud2(cloud, pcl_point_cloud_2);
  // Convert to sensor_msgs::PointCloud2.
  pcl_conversions::fromPCL(pcl_point_cloud_2, *converted);
  // Apply frame to msg.
  converted->header.frame_id = frame;
}

static void convert_to_point_cloud_2_msg(const segmatch::PointCloud& cloud,
                                         const std::string& frame,
                                         sensor_msgs::PointCloud2* converted) {
  segmatch::PointICloud cloud_i;
  pcl::copyPointCloud(cloud, cloud_i);
  convert_to_point_cloud_2_msg(cloud_i, frame, converted);
}

static void publishCloud(const segmatch::PointICloud& cloud, const std::string& frame,
                         const ros::Publisher& publisher) {
  sensor_msgs::PointCloud2 cloud_as_message;
  convert_to_point_cloud_2_msg(cloud, frame, &cloud_as_message);
  publisher.publish(cloud_as_message);
}

static segmatch::SegMatchParams getSegMatchParams(const ros::NodeHandle& nh,
                                                  const std::string& prefix) {
  segmatch::SegMatchParams params;

  std::string ns = prefix + "/SegMatch";

  nh.getParam(ns + "/segmentation_radius_m",
              params.segmentation_radius_m);
  nh.getParam(ns + "/filter_boundary_segments",
              params.filter_boundary_segments);
  nh.getParam(ns + "/boundary_radius_m",
              params.boundary_radius_m);
  nh.getParam(ns + "/filter_duplicate_segments",
              params.filter_duplicate_segments);
  nh.getParam(ns + "/centroid_distance_threshold_m",
              params.centroid_distance_threshold_m);

  // Descriptors parameters.
  nh.getParam(ns + "/Descriptors/descriptor_types",
              params.descriptors_params.descriptor_types);
  nh.getParam(ns + "/Descriptors/fast_point_feature_histograms_search_radius",
              params.descriptors_params.fast_point_feature_histograms_search_radius);
  nh.getParam(ns + "/Descriptors/fast_point_feature_histograms_normals_search_radius",
              params.descriptors_params.
              fast_point_feature_histograms_normals_search_radius);
  nh.getParam(ns + "/Descriptors/point_feature_histograms_search_radius",
              params.descriptors_params.point_feature_histograms_search_radius);
  nh.getParam(ns + "/Descriptors/point_feature_histograms_normals_search_radius",
              params.descriptors_params.point_feature_histograms_normals_search_radius);
  nh.getParam(ns + "/Descriptors/autoencoder_python_env",
              params.descriptors_params.autoencoder_python_env);
  nh.getParam(ns + "/Descriptors/autoencoder_script_path",
              params.descriptors_params.autoencoder_script_path);
  nh.getParam(ns + "/Descriptors/autoencoder_model_path",
              params.descriptors_params.autoencoder_model_path);
  nh.getParam(ns + "/Descriptors/autoencoder_temp_folder_path",
              params.descriptors_params.autoencoder_temp_folder_path);
  nh.getParam(ns + "/Descriptors/autoencoder_latent_space_dimension",
              params.descriptors_params.autoencoder_latent_space_dimension);

  // Segmenter parameters.
  nh.getParam(ns + "/Segmenters/segmenter_type",
              params.segmenter_params.segmenter_type);
  nh.getParam(ns + "/Segmenters/don_segmenter_small_scale",
              params.segmenter_params.don_segmenter_small_scale);
  nh.getParam(ns + "/Segmenters/don_segmenter_large_scale",
              params.segmenter_params.don_segmenter_large_scale);
  nh.getParam(ns + "/Segmenters/don_segmenter_don_threshold",
              params.segmenter_params.don_segmenter_don_threshold);
  nh.getParam(ns + "/Segmenters/don_segmenter_distance_tolerance",
              params.segmenter_params.don_segmenter_distance_tolerance);

  nh.getParam(ns + "/Segmenters/rg_min_cluster_size",
              params.segmenter_params.rg_min_cluster_size);
  nh.getParam(ns + "/Segmenters/rg_max_cluster_size",
              params.segmenter_params.rg_max_cluster_size);
  nh.getParam(ns + "/Segmenters/rg_knn_for_normals",
              params.segmenter_params.rg_knn_for_normals);
  nh.getParam(ns + "/Segmenters/rg_radius_for_normals",
              params.segmenter_params.rg_radius_for_normals);
  nh.getParam(ns + "/Segmenters/rg_knn_for_growing",
              params.segmenter_params.rg_knn_for_growing);
  nh.getParam(ns + "/Segmenters/rg_smoothness_threshold_deg",
              params.segmenter_params.rg_smoothness_threshold_deg);
  nh.getParam(ns + "/Segmenters/rg_curvature_threshold",
              params.segmenter_params.rg_curvature_threshold);

  nh.getParam(ns + "/Segmenters/ec_tolerance",
              params.segmenter_params.ec_tolerance);
  nh.getParam(ns + "/Segmenters/ec_max_cluster_size",
              params.segmenter_params.ec_max_cluster_size);
  nh.getParam(ns + "/Segmenters/ec_min_cluster_size",
              params.segmenter_params.ec_min_cluster_size);

  // Classifier parameters.
  nh.getParam(ns + "/Classifier/classifier_filename",
              params.classifier_params.classifier_filename);
  nh.getParam(ns + "/Classifier/threshold_to_accept_match",
              params.classifier_params.threshold_to_accept_match);

  nh.getParam(ns + "/Classifier/rf_max_depth",
              params.classifier_params.rf_max_depth);
  nh.getParam(ns + "/Classifier/rf_min_sample_ratio",
              params.classifier_params.rf_min_sample_ratio);
  nh.getParam(ns + "/Classifier/rf_regression_accuracy",
              params.classifier_params.rf_regression_accuracy);
  nh.getParam(ns + "/Classifier/rf_use_surrogates",
              params.classifier_params.rf_use_surrogates);
  nh.getParam(ns + "/Classifier/rf_max_categories",
              params.classifier_params.rf_max_categories);
  nh.getParam(ns + "/Classifier/rf_priors",
              params.classifier_params.rf_priors);
  nh.getParam(ns + "/Classifier/rf_calc_var_importance",
              params.classifier_params.rf_calc_var_importance);
  nh.getParam(ns + "/Classifier/rf_n_active_vars",
              params.classifier_params.rf_n_active_vars);
  nh.getParam(ns + "/Classifier/rf_max_num_of_trees",
              params.classifier_params.rf_max_num_of_trees);
  nh.getParam(ns + "/Classifier/rf_accuracy",
              params.classifier_params.rf_accuracy);

  // Convenience copy to find the correct feature distance according to
  // descriptors types.
  nh.getParam(ns + "/Descriptors/descriptor_types",
              params.classifier_params.descriptor_types);

  nh.getParam(ns + "/Classifier/n_nearest_neighbours",
              params.classifier_params.n_nearest_neighbours);
  nh.getParam(ns + "/Classifier/enable_two_stage_retrieval",
              params.classifier_params.enable_two_stage_retrieval);
  nh.getParam(ns + "/Classifier/knn_feature_dim",
              params.classifier_params.knn_feature_dim);
  nh.getParam(ns + "/Classifier/apply_hard_threshold_on_feature_distance",
              params.classifier_params.apply_hard_threshold_on_feature_distance);
  nh.getParam(ns + "/Classifier/feature_distance_threshold",
              params.classifier_params.feature_distance_threshold);

  nh.getParam(ns + "/Classifier/normalize_eigen_for_knn",
              params.classifier_params.normalize_eigen_for_knn);
  nh.getParam(ns + "/Classifier/normalize_eigen_for_hard_threshold",
              params.classifier_params.normalize_eigen_for_hard_threshold);
  nh.getParam(ns + "/Classifier/max_eigen_features_values",
              params.classifier_params.max_eigen_features_values);


  // Geometric Consistency Parameters.
  nh.getParam(ns + "/GeometricConsistency/resolution",
              params.geometric_consistency_params.resolution);
  nh.getParam(ns + "/GeometricConsistency/min_cluster_size",
              params.geometric_consistency_params.min_cluster_size);

  return params;
}

static SegMatchWorkerParams getSegMatchWorkerParams(const ros::NodeHandle& nh,
                                                    const std::string& prefix) {
  SegMatchWorkerParams params;
  const std::string ns = prefix + "/SegMatchWorker";
  nh.getParam(ns + "/localize", params.localize);
  nh.getParam(ns + "/close_loops", params.close_loops);

  if (params.localize && params.close_loops) {
    LOG(INFO) << "Parameters localize and close_loops both set.";
    LOG(INFO) << "Setting close_loops to false.";
    params.close_loops = false;
  }

  if (params.localize) {
    using namespace boost::filesystem;
    nh.getParam(ns + "/target_cloud_filename", params.target_cloud_filename);
    path target_cloud_path(params.target_cloud_filename);
    CHECK(exists(target_cloud_path)) << "Target cloud does not exist.";
  }

  nh.getParam(ns +"/distance_between_segmentations_m",
              params.distance_between_segmentations_m);

  nh.getParam(ns +"/distance_to_lower_target_cloud_for_viz_m",
              params.distance_to_lower_target_cloud_for_viz_m);

  nh.getParam(ns +"/align_target_map_on_first_loop_closure",
              params.align_target_map_on_first_loop_closure);

  nh.getParam(ns +"/export_segments_and_matches",
              params.export_segments_and_matches);

  nh.getParam(ns +"/ratio_of_points_to_keep_when_publishing",
              params.ratio_of_points_to_keep_when_publishing);

  params.segmatch_params = getSegMatchParams(nh, ns);

  return params;
}

// TODO needed?
static segmatch::GroundTruthParameters getGroundTruthParams(
    const ros::NodeHandle& nh, const std::string& prefix) {
  segmatch::GroundTruthParameters params;
  nh.getParam(prefix + "/GroundTruth/overlap_radius", params.overlap_radius);
  nh.getParam(prefix + "/GroundTruth/significance_percentage", params.significance_percentage);
  nh.getParam(prefix + "/GroundTruth/number_nearest_segments", params.number_nearest_segments);
  nh.getParam(prefix + "/GroundTruth/maximum_centroid_distance_m",
              params.maximum_centroid_distance_m);
  return params;
}

static segmatch::PclPoint tfTransformToPoint(const tf::StampedTransform& tf_transform) {
  segmatch::PclPoint point;
  point.x = tf_transform.getOrigin().getX();
  point.y = tf_transform.getOrigin().getY();
  point.z = tf_transform.getOrigin().getZ();
  return point;
}

segmatch::SE3 geometryMsgTransformToSE3(const geometry_msgs::Transform& transform) {
  segmatch::SE3::Position pos(transform.translation.x, transform.translation.y,
                              transform.translation.z);
  segmatch::SE3::Rotation::Implementation rot(transform.rotation.w, transform.rotation.x,
                                              transform.rotation.y, transform.rotation.z);
  return segmatch::SE3(pos, rot);
}

geometry_msgs::Transform SE3ToGeometryMsgTransform(const segmatch::SE3& transform) {
  geometry_msgs::Transform result;
  Eigen::Affine3d eigen_transform(transform.getTransformationMatrix());
  result.translation.x = eigen_transform(0,3);
  result.translation.y = eigen_transform(1,3);
  result.translation.z = eigen_transform(2,3);
  Eigen::Quaterniond rotation(eigen_transform.rotation());
  result.rotation.w = rotation.w();
  result.rotation.x = rotation.x();
  result.rotation.y = rotation.y();
  result.rotation.z = rotation.z();
  // TODO: safe casting double to float?
  return result;
}

/// \brief Covariance type including pose and uncertainty information.
struct CovarianceMsg {
  /// \brief Pose vector.
  geometry_msgs::Pose pose;
  /// \brief Uncertainty vector.
  geometry_msgs::Vector3 magnitude;
};
typedef std::vector<CovarianceMsg> CovarianceMsgs;

// Convert SE3 object to ROS geometry message pose.
static void convert_to_geometry_msg_pose(const segmatch::SE3& pose,
                                         geometry_msgs::Pose* pose_msg) {
  CHECK_NOTNULL(pose_msg);
  pose_msg->position.x = pose.getPosition().x();
  pose_msg->position.y = pose.getPosition().y();
  pose_msg->position.z = pose.getPosition().z();
  pose_msg->orientation.w = pose.getRotation().w();
  pose_msg->orientation.x = pose.getRotation().x();
  pose_msg->orientation.y = pose.getRotation().y();
  pose_msg->orientation.z = pose.getRotation().z();
}

// Publish ellipsoids given covariance matrices and fixed frame.
static void drawCovarianceEllipsoids(const std::string& frame,
                                     const CovarianceMsgs& covariances,
                                     const ros::Publisher& publisher) {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;

  for (size_t i = 0u; i < covariances.size(); ++i) {
    marker.header.frame_id = frame;
    marker.header.stamp = ros::Time();
    marker.ns = "laser_mapper";
    marker.id = i;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose = covariances.at(i).pose;
    marker.scale = covariances.at(i).magnitude;
    marker.color.a = 1.0;

    // Distinguish start pose covariance.
    if (i == 0u) {
      marker.color.r = 1.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
    } else {
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.2;
    }

    // Keep marker until updated.
    marker.lifetime = ros::Duration();

    marker_array.markers.push_back(marker);
  }
  publisher.publish(marker_array);
}

} // namespace segmatch_ros

#endif // SEGMATCH_ROS_COMMON_HPP_
