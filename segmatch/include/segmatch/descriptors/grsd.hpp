#ifndef SEGMATCH_GRSD_HPP_
#define SEGMATCH_GRSD_HPP_

#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

class GrsdDescriptor : public Descriptor {
 public:
  GrsdDescriptor () {};
  explicit GrsdDescriptor(const DescriptorsParameters& parameters);
  ~GrsdDescriptor () {};

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

  /// \brief Get the descriptor's dimension.
  virtual unsigned int dimension() const { return kDimension; };

  virtual void exportData() const { };

 private:
  static constexpr unsigned int kDimension = 8u;
}; // class GrsdDescriptor

} // namespace segmatch

#endif // SEGMATCH_GRSD_HPP_
