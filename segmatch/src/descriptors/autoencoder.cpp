#include "segmatch/descriptors/autoencoder.hpp"

#include <stdlib.h> /* system, NULL, EXIT_FAILURE */

#include <Eigen/Core>
#include <glog/logging.h>

#include "segmatch/database.hpp"

namespace segmatch {

void AutoencoderDescriptor::describe(const Segment& segment, Features* features) {
  SegmentedCloud single_segment_segmented_cloud;
  single_segment_segmented_cloud.addValidSegment(segment);

  describe(&single_segment_segmented_cloud);

  *features = single_segment_segmented_cloud.getValidSegmentByIndex(0).features;
}

void AutoencoderDescriptor::describe(SegmentedCloud* segmented_cloud_ptr) {
  CHECK_NOTNULL(segmented_cloud_ptr);
  // TODO: get rid of kDimension.
  CHECK(kDimension == params_.autoencoder_latent_space_dimension + 3) << "kDimension != params."; 

  // Export segmented cloud.
  export_segments(params_.autoencoder_temp_folder_path + kSegmentsFilename, *segmented_cloud_ptr);

  // Wait for script to describe segments.
  char buff[512];
  while (fgets(buff, sizeof(buff), script_process_pipe_)!=NULL) {
    LOG(INFO) << buff;
    if (std::string(buff) == "__DESC_COMPLETE__\n") {
      break;
    }
  }

  // Import the autoencoder features from file.
  LOG(INFO) << "Importing autoencoder features";
  CHECK(import_features(params_.autoencoder_temp_folder_path + kFeaturesFilename,
                        segmented_cloud_ptr,
                        "concatenate"));
  LOG(INFO) << "Done.";
}

} // namespace segmatch
