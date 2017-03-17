#ifndef SEGMATCH_AUTOENCODER_HPP_
#define SEGMATCH_AUTOENCODER_HPP_

#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

class AutoencoderDescriptor : public Descriptor {
 public:
  //AutoencoderDescriptor () {};
  explicit AutoencoderDescriptor(const DescriptorsParameters& parameters) : params_(parameters) {
    // Call the autoencoder python script.
    const std::string command = params_.autoencoder_python_env + " -u " +
        params_.autoencoder_script_path + " " +
        params_.autoencoder_model_path + " " +
        params_.autoencoder_temp_folder_path + kSegmentsFilename + " " +
        params_.autoencoder_temp_folder_path + kFeaturesFilename + " " +
        std::to_string(params_.autoencoder_latent_space_dimension);
    LOG(INFO) << "Executing command: $" << command;
    if (!(script_process_pipe_ = popen(command.c_str(), "r"))) {
      LOG(FATAL) << "Could not execute autoencoder command";
    }
    char buff[512];
    while (fgets(buff, sizeof(buff), script_process_pipe_) != NULL) {
      LOG(INFO) << buff;
      if (std::string(buff) == "__INIT_COMPLETE__\n") {
        break;
      }
    }
    LOG(INFO) << "Autoencoder script initialization complete.";
  }

  ~AutoencoderDescriptor () {
    pclose(script_process_pipe_);
  };

  // Use methods common to all descriptor children.
  using Descriptor::describe;

  /// \brief Describe the segment by modifying a Features object.
  virtual void describe(const Segment& segment, Features* features);

  /// \brief Overrides the normal segmented cloud iterative description to be more efficient.
  virtual void describe(SegmentedCloud* segmented_cloud_ptr);

  /// \brief Get the descriptor's dimension.
  virtual unsigned int dimension() const { return kDimension; };

 private:
  // TODO: the dimension is unknown.
  static constexpr unsigned int kDimension = 22u;

  DescriptorsParameters params_;

  FILE* script_process_pipe_;
  const std::string kSegmentsFilename = "autoencoder_segments.txt";
  const std::string kFeaturesFilename = "autoencoder_features.txt";
}; // class AutoencoderDescriptor

} // namespace segmatch

#endif // SEGMATCH_AUTOENCODER_HPP_
