#ifndef SEGMATCH_ENSEMBLE_SHAPE_FUNCTIONS_HPP_
#define SEGMATCH_ENSEMBLE_SHAPE_FUNCTIONS_HPP_

#include <pcl/features/esf.h>

#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

class EnsembleShapeFunctions : public Descriptor {
 public:
  EnsembleShapeFunctions () {};
  explicit EnsembleShapeFunctions(const DescriptorsParameters& parameters) {};
  ~EnsembleShapeFunctions () {};

  // Use methods common to all descriptor children.
  using Descriptor::describe;

  virtual void describe(const Segment& segment, Features* features);

  virtual void describe(SegmentedCloud* segmented_cloud_ptr) {
    CHECK_NOTNULL(segmented_cloud_ptr);
    for (std::unordered_map<Id, Segment>::iterator it = segmented_cloud_ptr->begin();
        it != segmented_cloud_ptr->end(); ++it) {
      describe(&(it->second));
    }
  }

  unsigned int dimension() const { return kSignatureDimension; };

  virtual void exportData() const { };

 private:
  pcl::ESFEstimation<PclPoint, pcl::ESFSignature640> esf_estimator_;

  static constexpr unsigned int kSignatureDimension = 640u;
}; // class EnsembleShapeFunctions

} // namespace segmatch

#endif // SEGMATCH_ENSEMBLE_SHAPE_FUNCTIONS_HPP_
