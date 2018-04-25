#ifndef SEGMATCH_DESCRIPTORS_HPP_
#define SEGMATCH_DESCRIPTORS_HPP_

#include <memory>
#include <string>

#include "segmatch/features.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

class Descriptor {
 public:
  Descriptor();
  ~Descriptor();
  /// \brief Describe the segment by modifying its internal Features object.
  void describe(Segment* segment_ptr);

  /// \brief Describe the segment by modifying a Features object.
  virtual void describe(const Segment& segment, Features* features) = 0;

  /// \brief Describe the whole segmented cloud at once.
  virtual void describe(SegmentedCloud* segmented_cloud_ptr) = 0;

  /// \brief Get the descriptor's dimension.
  virtual unsigned int dimension() const = 0;

  virtual void exportData() const = 0;
}; // class Descriptor

class Descriptors {
 public:
  Descriptors();
  explicit Descriptors(const DescriptorsParameters& parameters);
  ~Descriptors();

  /// \brief Describe the segment by modifying its internal Features object.
  void describe(Segment* segment_ptr);

  /// \brief Describe the segment by modifying a Features object.
  void describe(const Segment& segment, Features* features);

  /// \brief Describe all the segments in a segmented cloud with all the descriptors.
  void describe(SegmentedCloud* segmented_cloud_ptr);

  /// \brief Get the total dimension of the descriptors.
  unsigned int dimension() const;

  /// \brief Export descriptors related data.
  void exportData() const {
    for (const auto& descriptor : descriptors_) descriptor->exportData();
  };

 private:
  std::vector<std::unique_ptr<Descriptor> > descriptors_;
}; // class Descriptors

} // namespace segmatch

#endif // SEGMATCH_DESCRIPTORS_HPP_
