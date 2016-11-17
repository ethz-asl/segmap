#ifndef SEGMATCH_EIGENVALUE_BASED_HPP_
#define SEGMATCH_EIGENVALUE_BASED_HPP_

#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

class EigenvalueBasedDescriptor : public Descriptor {
 public:
  EigenvalueBasedDescriptor () {};
  explicit EigenvalueBasedDescriptor(const DescriptorsParameters& parameters);
  ~EigenvalueBasedDescriptor () {};

  // Use methods common to all descriptor children.
  using Descriptor::describe;

  virtual void describe(const Segment& segment, Features* features);

  virtual void describe(SegmentedCloud* segmented_cloud_ptr) {
    CHECK_NOTNULL(segmented_cloud_ptr);
    for (size_t i = 0u; i < segmented_cloud_ptr->getNumberOfValidSegments(); ++i) {
      describe(segmented_cloud_ptr->getValidSegmentPtrByIndex(i));
    }
  }

  /// \brief Get the descriptor's dimension.
  virtual unsigned int dimension() const { return kDimension; };

 private:
  static constexpr unsigned int kDimension = 7u;
}; // class EigenvalueBasedDescriptor

} // namespace segmatch

#endif // SEGMATCH_EIGENVALUE_BASED_HPP_
