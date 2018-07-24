#include "segmatch/normal_estimators/normal_estimator.hpp"

#include <glog/logging.h>

#include "segmatch/normal_estimators/incremental_normal_estimator.hpp"
#include "segmatch/normal_estimators/simple_normal_estimator.hpp"

namespace segmatch {

std::unique_ptr<NormalEstimator> NormalEstimator::create(
    const std::string& estimator_type, const float radius_for_estimation_m) {
  if (estimator_type == "Simple") {
    return std::unique_ptr<NormalEstimator>(
        new SimpleNormalEstimator(radius_for_estimation_m));
  } else if (estimator_type == "Incremental") {
    return std::unique_ptr<NormalEstimator>(
        new IncrementalNormalEstimator(radius_for_estimation_m));
  } else {
    LOG(FATAL) << "Invalid normal estimator type specified: " << estimator_type;
    throw std::invalid_argument("Invalid normal estimator type specified: " + estimator_type);
  }
}

} // namespace segmatch
