#ifndef SEGMATCH_CNN_HPP_
#define SEGMATCH_CNN_HPP_

#include <laser_slam/common.hpp>
#include <tf_graph_executor/tf_graph_executor.hpp>

#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

class CNNDescriptor : public Descriptor {
 public:
  //AutoencoderDescriptor () {};
  explicit CNNDescriptor(const DescriptorsParameters& parameters) : params_(parameters) {
    const std::string model_folder = parameters.cnn_model_path;
    const std::string semantics_nn_folder = parameters.semantics_nn_path;

    LOG(INFO) << "Loading CNN model in " + model_folder;
    graph_executor_.reset(new tf_graph_executor::TensorflowGraphExecutor(
        model_folder + "model.ckpt.meta"));
    graph_executor_->loadCheckpoint(model_folder + "model.ckpt");

    aligned_segments_ = SegmentedCloud(false);

    /*Eigen::MatrixXd voxel_mean_values;
    laser_slam::loadEigenMatrixXdCSV(model_folder + "scaler_mean.csv", &voxel_mean_values);

    CHECK_EQ(voxel_mean_values.rows(), cnn_input_dim_);
    CHECK_EQ(voxel_mean_values.cols(), 1u);

    for (size_t i = 0u; i < cnn_input_dim_; ++i) {
      voxel_mean_values_.push_back(voxel_mean_values(i, 0u));
    }*/

    // Load the semantics nn model.
    LOG(INFO) << "Loading semantics model in " + semantics_nn_folder;
    semantics_graph_executor_.reset(new tf_graph_executor::TensorflowGraphExecutor(
        semantics_nn_folder + "model.ckpt.meta"));
    semantics_graph_executor_->loadCheckpoint(semantics_nn_folder + "model.ckpt");

    LOG(INFO) << "Loaded all TensorFlow models.";
  }

  ~CNNDescriptor () {};

  // Use methods common to all descriptor children.
  using Descriptor::describe;

  /// \brief Describe the segment by modifying a Features object.
  virtual void describe(const Segment& segment, Features* features);

  /// \brief Overrides the normal segmented cloud iterative description to be more efficient.
  virtual void describe(SegmentedCloud* segmented_cloud_ptr);

  /// \brief Get the descriptor's dimension.
  virtual unsigned int dimension() const { return kDimension; };

  virtual void exportData() const;

 private:
  // TODO: the dimension is unknown.
  static constexpr unsigned int kDimension = 32u;

  DescriptorsParameters params_;

  std::shared_ptr<tf_graph_executor::TensorflowGraphExecutor> graph_executor_;
  std::shared_ptr<tf_graph_executor::TensorflowGraphExecutor> semantics_graph_executor_;

  std::vector<float> voxel_mean_values_;

  segmatch::SegmentedCloud aligned_segments_;

  constexpr static float min_voxel_size_m_ = 0.1;
  
  constexpr static unsigned int n_voxels_x_dim_ = 32u;
  constexpr static unsigned int n_voxels_y_dim_ = 32u;
  constexpr static unsigned int n_voxels_z_dim_ = 16u;
  constexpr static unsigned int cnn_input_dim_ = n_voxels_x_dim_ * n_voxels_y_dim_ *
      n_voxels_z_dim_;
      
  constexpr static float min_x_scale_m_ = static_cast<float>(n_voxels_x_dim_) * min_voxel_size_m_;
  constexpr static float min_y_scale_m_ = static_cast<float>(n_voxels_y_dim_) * min_voxel_size_m_;
  constexpr static float min_z_scale_m_ = static_cast<float>(n_voxels_z_dim_) * min_voxel_size_m_;
  
  constexpr static float x_dim_min_1_ = static_cast<float>(n_voxels_x_dim_) - 1.0;
  constexpr static float y_dim_min_1_ = static_cast<float>(n_voxels_y_dim_) - 1.0;
  constexpr static float z_dim_min_1_ = static_cast<float>(n_voxels_z_dim_) - 1.0;

  constexpr static size_t mini_batch_size_ = 10u;

  constexpr static bool save_debug_data_ = true;

  const std::string kInputTensorName = "InputScope/input";
  const std::string kFeaturesTensorName = "OutputScope/descriptor_read";
  const std::string kSemanticsOutputName = "OutputScope/output_read";
  const std::string kReconstructionTensorName = "ReconstructionScopeAE/ae_reconstruction_read";
  const std::string kScalesTensorName = "scales";

}; // class CNNDescriptor

} // namespace segmatch

#endif // SEGMATCH_CNN_HPP_
