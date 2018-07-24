#include "segmatch/recognizers/correspondence_recognizer_factory.hpp"

#include "segmatch/recognizers/geometric_consistency_recognizer.hpp"
#include "segmatch/recognizers/incremental_geometric_consistency_recognizer.hpp"
#include "segmatch/recognizers/partitioned_geometric_consistency_recognizer.hpp"

namespace segmatch {

CorrespondenceRecognizerFactory::CorrespondenceRecognizerFactory(const SegMatchParams& params)
  : params_(params.geometric_consistency_params),
    local_map_radius_(params.local_map_params.radius_m) {
}

std::unique_ptr<CorrespondenceRecognizer> CorrespondenceRecognizerFactory::create() const {
  if (params_.recognizer_type == "Simple") {
    return std::unique_ptr<CorrespondenceRecognizer>(
        new GeometricConsistencyRecognizer(params_));
  } else if (params_.recognizer_type == "Partitioned") {
    return std::unique_ptr<CorrespondenceRecognizer>(
        new PartitionedGeometricConsistencyRecognizer(params_, local_map_radius_));
  } else if (params_.recognizer_type == "Incremental") {
    return std::unique_ptr<CorrespondenceRecognizer>(
        new IncrementalGeometricConsistencyRecognizer(params_, local_map_radius_));
  } else {
    LOG(FATAL) << "Invalid recognizer type specified: " << params_.recognizer_type;
    throw std::invalid_argument("Invalid recognizer type specified: " + params_.recognizer_type);
  }
}

} // namespace segmatch
