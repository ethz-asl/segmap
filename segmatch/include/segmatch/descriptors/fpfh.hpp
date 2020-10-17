#ifndef SEGMATCH_FPFH_HPP_
#define SEGMATCH_FPFH_HPP_

#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

class FpfhDescriptor : public Descriptor {
 public:
  FpfhDescriptor () {};
  explicit FpfhDescriptor(const DescriptorsParameters& parameters);
  ~FpfhDescriptor () {};

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
}; // class FpfhDescriptor

} // namespace segmatch

#endif // SEGMATCH_FPFH_HPP_
