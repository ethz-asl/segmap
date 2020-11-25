#include "segmatch/descriptors/cnn.hpp"

#include <glog/logging.h>
#include <cmath>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <laser_slam/benchmarker.hpp>
#include <stdlib.h>
#include <Eigen/Core>
#include <algorithm>
#include <cstdio>
#include <string>

#include "segmatch/database.hpp"
#include "segmatch/utilities.hpp"

namespace segmatch {

CNNDescriptor::CNNDescriptor(const DescriptorsParameters& parameters)
    : params_(parameters) {
  const std::string model_folder = parameters.cnn_model_path;
  const std::string semantics_nn_folder = parameters.semantics_nn_path;

  interface_worker_ = std::unique_ptr<TensorflowInterface>(
      new TensorflowInterface());
}

void CNNDescriptor::describe(const Segment& segment, Features* features) {
  CHECK(false) << "Not implemented";
}

template <typename T>
std::vector<size_t> getIndexesInDecreasingOrdering(const std::vector<T>& v) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

void CNNDescriptor::describe(SegmentedCloud* segmented_cloud_ptr) {
  CHECK_NOTNULL(segmented_cloud_ptr);
  constexpr double kMinChangeBeforeDescription = 0.1;  // 0.2

  BENCHMARK_START("SM.Worker.Describe.Preprocess");
  BENCHMARK_RECORD_VALUE("SM.Worker.Describe.NumSegmentsTotal",
                         segmented_cloud_ptr->getNumberOfValidSegments());
  std::vector<VoxelPointCloud> batch_nn_input;
  std::vector<Id> described_segment_ids;
  std::vector<PclPoint> scales;
  std::vector<PclPoint> thresholded_scales;
  std::vector<std::vector<float> > scales_as_vectors;
  std::vector<PclPoint> rescaled_point_cloud_centroids;
  std::vector<PclPoint> point_mins;
  std::vector<double> alignments_rad;

  for (std::unordered_map<Id, Segment>::iterator it =
           segmented_cloud_ptr->begin();
       it != segmented_cloud_ptr->end(); ++it) {
    const PointCloud& point_cloud = it->second.getLastView().point_cloud;
    const size_t num_points = point_cloud.size();

    // Skip describing the segment if it did not change enough.
    if (static_cast<double>(num_points) <
        static_cast<double>(
            it->second.getLastView().n_points_when_last_described) *
            (1.0 + kMinChangeBeforeDescription)) {
      continue;
    }

    it->second.getLastView().n_points_when_last_described = num_points;
    described_segment_ids.push_back(it->second.segment_id);

    // Align with PCA.
    double alignment_rad;
    Eigen::Vector4f pca_centroid;
    pcl::compute3DCentroid(point_cloud, pca_centroid);
    Eigen::Matrix3f covariance_3d;
    computeCovarianceMatrixNormalized(point_cloud, pca_centroid, covariance_3d);
    const Eigen::Matrix2f covariance_2d = covariance_3d.block(0, 0, 2u, 2u);
    Eigen::EigenSolver<Eigen::Matrix2f> eigen_solver(covariance_2d, true);

    alignment_rad = atan2(eigen_solver.eigenvectors()(1, 0).real(),
                          eigen_solver.eigenvectors()(0, 0).real());

    if (eigen_solver.eigenvalues()(0).real() <
        eigen_solver.eigenvalues()(1).real()) {
      alignment_rad += 0.5 * M_PI;
    }

    // Rotate the segment.
    alignment_rad = -alignment_rad;
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(
        Eigen::AngleAxisf(alignment_rad, Eigen::Vector3f::UnitZ()));
    PointCloud rotated_point_cloud;
    pcl::transformPointCloud(point_cloud, rotated_point_cloud, transform);

    // Get most points on the lower half of y axis (by rotation).
    PclPoint point_min, point_max;
    pcl::getMinMax3D(rotated_point_cloud, point_min, point_max);
    double centroid_y = point_min.y + (point_max.y - point_min.y) / 2.0;
    uint32_t n_below = 0;
    for (const auto& point : rotated_point_cloud.points) {
      if (point.y < centroid_y) ++n_below;
    }
    if (static_cast<double>(n_below) <
        static_cast<double>(rotated_point_cloud.size()) / 2.0) {
      alignment_rad += M_PI;
      Eigen::Affine3f adjustment_transform = Eigen::Affine3f::Identity();
      adjustment_transform.rotate(
          Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));
      pcl::transformPointCloud(rotated_point_cloud, rotated_point_cloud,
                               adjustment_transform);
    }

    alignments_rad.push_back(alignment_rad);

    PointCloud rescaled_point_cloud;
    pcl::getMinMax3D(rotated_point_cloud, point_min, point_max);
    point_mins.push_back(point_min);

    // "Fit scaling" using the largest dimension as scale.
    PclPoint scale;
    scale.x = point_max.x - point_min.x;
    scale.y = point_max.y - point_min.y;
    scale.z = point_max.z - point_min.z;
    scales.push_back(scale);

    PclPoint thresholded_scale;
    thresholded_scale.x = std::max(scale.x, min_x_scale_m_);
    thresholded_scale.y = std::max(scale.y, min_y_scale_m_);
    thresholded_scale.z = std::max(scale.z, min_z_scale_m_);
    thresholded_scales.push_back(thresholded_scale);

    std::vector<float> scales_as_vector;
    scales_as_vector.push_back(scale.x);
    scales_as_vector.push_back(scale.y);
    scales_as_vector.push_back(scale.z);
    scales_as_vectors.push_back(scales_as_vector);

    for (const auto& point : rotated_point_cloud.points) {
      PclPoint point_new;

      point_new.x = (point.x - point_min.x) / thresholded_scale.x *
                    static_cast<float>(n_voxels_x_dim_ - 1u);
      point_new.y = (point.y - point_min.y) / thresholded_scale.y *
                    static_cast<float>(n_voxels_y_dim_ - 1u);
      point_new.z = (point.z - point_min.z) / thresholded_scale.z *
                    static_cast<float>(n_voxels_z_dim_ - 1u);
      point_new.rgba = point.rgba;

      rescaled_point_cloud.points.push_back(point_new);
    }

    rescaled_point_cloud.width = 1;
    rescaled_point_cloud.height = rescaled_point_cloud.points.size();

    PclPoint centroid = calculateCentroid(rescaled_point_cloud);
    rescaled_point_cloud_centroids.push_back(centroid);

    VoxelPointCloud voxel_point_cloud;
    for (const auto& point : rescaled_point_cloud.points) {
      const uint32_t ind_x = floor(
          point.x + static_cast<float>(n_voxels_x_dim_ - 1) / 2.0 - centroid.x);
      const uint32_t ind_y = floor(
          point.y + static_cast<float>(n_voxels_y_dim_ - 1) / 2.0 - centroid.y);
      const uint32_t ind_z = floor(
          point.z + static_cast<float>(n_voxels_z_dim_ - 1) / 2.0 - centroid.z);

      if (ind_x >= 0 && ind_x < n_voxels_x_dim_ && ind_y >= 0 &&
          ind_y < n_voxels_y_dim_ && ind_z >= 0 && ind_z < n_voxels_z_dim_) {
        VoxelPoint voxel_point;
        voxel_point.x = ind_x;
        voxel_point.y = ind_y;
        voxel_point.z = ind_z;
        voxel_point.r = (point.rgba >> 16) & 0xff;
        voxel_point.g = (point.rgba >> 8) & 0xff;
        voxel_point.b = point.rgba & 0xff;
        voxel_point.semantic_class = ((point.rgba >> 24) & 0xff) / 7;

        voxel_point_cloud.push_back(voxel_point);
      }
    }

    batch_nn_input.push_back(voxel_point_cloud);
  }
  BENCHMARK_RECORD_VALUE("SM.Worker.Describe.NumSegmentsDescribed",
                         batch_nn_input.size());
  BENCHMARK_STOP("SM.Worker.Describe.Preprocess");

  std::cout << "PROCESSING SEGMENTS " << batch_nn_input.size() << std::endl;
  if (!batch_nn_input.empty()) {
    BENCHMARK_START("SM.Worker.Describe.ForwardPass");
    std::vector<std::vector<float>> cnn_descriptors;
    std::vector<PointCloud> reconstructions;
    std::vector<std::vector<float>> semantics;
    if (batch_nn_input.size() < mini_batch_size_) {

      interface_worker_->batchFullForwardPass(
          batch_nn_input, kInputTensorName, scales_as_vectors,
          kScalesTensorName, kFeaturesTensorName, kReconstructionTensorName,
          cnn_descriptors, reconstructions);


    } else {
      std::vector<VoxelPointCloud> mini_batch;
      std::vector<std::vector<float> > mini_batch_scales;
      for (size_t i = 0u; i < batch_nn_input.size(); ++i) {
        if ((i+1) % 100 == 0) {
          std::cout << "Progress " << i << std::endl;
        }

        mini_batch.push_back(batch_nn_input[i]);
        mini_batch_scales.push_back(scales_as_vectors[i]);

        if (mini_batch.size() == mini_batch_size_ || i == batch_nn_input.size() - 1) {
          std::vector<std::vector<float> > mini_batch_cnn_descriptors;
          std::vector<PointCloud> mini_batch_reconstructions;

          interface_worker_->batchFullForwardPass(
              mini_batch, kInputTensorName, mini_batch_scales,
              kScalesTensorName, kFeaturesTensorName, kReconstructionTensorName,
              mini_batch_cnn_descriptors, mini_batch_reconstructions);

          cnn_descriptors.insert(cnn_descriptors.end(),
                                 mini_batch_cnn_descriptors.begin(),
                                 mini_batch_cnn_descriptors.end());
          reconstructions.insert(reconstructions.end(),
                                 mini_batch_reconstructions.begin(),
                                 mini_batch_reconstructions.end());

          mini_batch_scales.clear();
          mini_batch.clear();
        }
      }
    }

    // Execute semantics graph.
    semantics = interface_worker_->batchExecuteGraph(
        cnn_descriptors, kInputTensorName, kSemanticsOutputName);

    CHECK_EQ(cnn_descriptors.size(), described_segment_ids.size());
    BENCHMARK_STOP("SM.Worker.Describe.ForwardPass");

    // Write the features.
    BENCHMARK_START("SM.Worker.Describe.SaveFeatures");
    for (size_t i = 0u; i < described_segment_ids.size(); ++i) {
      Segment* segment;
      CHECK(segmented_cloud_ptr->findValidSegmentPtrById(
          described_segment_ids[i], &segment));
      std::vector<float> nn_output = cnn_descriptors[i];

      Feature cnn_feature("cnn");
      for (size_t j = 0u; j < nn_output.size(); ++j) {
        cnn_feature.push_back(
            FeatureValue("cnn_" + std::to_string(j), nn_output[j]));
      }

      // Push the scales.
      cnn_feature.push_back(FeatureValue("cnn_scale_x", scales[i].x));
      cnn_feature.push_back(FeatureValue("cnn_scale_y", scales[i].y));
      cnn_feature.push_back(FeatureValue("cnn_scale_z", scales[i].z));

      segment->getLastView().features.replaceByName(cnn_feature);

      std::vector<float> semantic_nn_output = semantics[i];
      segment->getLastView().semantic =
          std::distance(semantic_nn_output.begin(),
                        std::max_element(semantic_nn_output.begin(),
                                         semantic_nn_output.end()));

      // Generate the reconstructions.
      /*PointCloud reconstruction;
      const double reconstruction_threshold = 0.75;

      PclPoint point;
      const PclPoint point_min = point_mins[i];
      const PclPoint scale = thresholded_scales[i];
      const PclPoint centroid = rescaled_point_cloud_centroids[i];

      for (uint32_t x = 0u; x < n_voxels_x_dim_; ++x) {
        for (uint32_t y = 0u; y < n_voxels_y_dim_; ++y) {
          for (uint32_t z = 0u; z < n_voxels_z_dim_; ++z) {
            if (reconstructions[i].at(x, y, z) >= reconstruction_threshold) {
              point.x = point_min.x + scale.x * (static_cast<float>(x) -
                  x_dim_min_1_ / 2.0 + centroid.x) / x_dim_min_1_;
              point.y = point_min.y + scale.y * (static_cast<float>(y) -
                  y_dim_min_1_ / 2.0 + centroid.y) / y_dim_min_1_;
              point.z = point_min.z + scale.z * (static_cast<float>(z) -
                  z_dim_min_1_ / 2.0 + centroid.z) / z_dim_min_1_;
              reconstruction.points.push_back(point);
            }
          }
        }
      }

      reconstruction.width = 1;
      reconstruction.height = reconstruction.points.size();

      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.rotate(
          Eigen::AngleAxisf(-alignments_rad[i], Eigen::Vector3f::UnitZ()));
      pcl::transformPointCloud(reconstruction, reconstruction, transform);
      segment->getLastView().reconstruction = reconstruction;*/
    }
    BENCHMARK_STOP("SM.Worker.Describe.SaveFeatures");
  }
}

void CNNDescriptor::exportData() const {}

}  // namespace segmatch
