#ifndef SEGMATCH_CNN_HPP_
#define SEGMATCH_CNN_HPP_

#include <laser_slam/common.hpp>
#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmented_cloud.hpp"
#include "segmatch/descriptors/tf_interface.hpp"

namespace segmatch {

class CNNDescriptor : public Descriptor {
 public:
  explicit CNNDescriptor(const DescriptorsParameters& parameters);
  ~CNNDescriptor () {};

  // Use methods common to all descriptor children.
  using Descriptor::describe;

  /// \brief Describe the segment by modifying a Features object.
  virtual void describe(const Segment& segment, Features* features);

  /// \brief Overrides the normal segmented cloud iterative description to be more efficient.
  virtual void describe(SegmentedCloud* segmented_cloud_ptr);

  /// \brief Get the descriptor's dimension.
  virtual unsigned dimension() const { return kDimension; };

  virtual void exportData() const;

 private:
  // TODO: the dimension is unknown.
  const unsigned kDimension = 32u;

  const unsigned n_voxels_x_dim_ = 32u;
  const unsigned n_voxels_y_dim_ = 32u;
  const unsigned n_voxels_z_dim_ = 16u;

  const float min_voxel_size_m_ = 0.1;
  const float min_x_scale_m_ = static_cast<float>(n_voxels_x_dim_) * min_voxel_size_m_;
  const float min_y_scale_m_ = static_cast<float>(n_voxels_y_dim_) * min_voxel_size_m_;
  const float min_z_scale_m_ = static_cast<float>(n_voxels_z_dim_) * min_voxel_size_m_;

  const float x_dim_min_1_ = static_cast<float>(n_voxels_x_dim_) - 1.0;
  const float y_dim_min_1_ = static_cast<float>(n_voxels_y_dim_) - 1.0;
  const float z_dim_min_1_ = static_cast<float>(n_voxels_z_dim_) - 1.0;

  const size_t mini_batch_size_ = 16u;

  const std::string kInputTensorName = "InputScope/input";
  const std::string kFeaturesTensorName = "OutputScope/descriptor_read";
  const std::string kSemanticsOutputName = "OutputScope/output_read";
  const std::string kReconstructionTensorName = "ReconstructionScopeAE/ae_reconstruction_read";
  const std::string kScalesTensorName = "scales";

  DescriptorsParameters params_;
  std::unique_ptr<TensorflowInterface> interface_worker_;
}; // class CNNDescriptor

} // namespace segmatch

#endif // SEGMATCH_CNN_HPP_
