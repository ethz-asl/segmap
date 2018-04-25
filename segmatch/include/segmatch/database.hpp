#ifndef SEGMATCH_DATABASE_HPP_
#define SEGMATCH_DATABASE_HPP_

#include <string>

#include "segmatch/segmented_cloud.hpp"

namespace segmatch {
namespace database {

/// \brief A structure for storing a position within a 2D table.
struct Position {
  size_t row = 0;
  size_t col = 0;
};

class UniqueIdMatches {
 public:
  UniqueIdMatches() {}
  explicit UniqueIdMatches(std::vector<std::vector<Id> > vector) { id_match_list_ = vector; }
  bool findMatches(const Id id, std::vector<Id>* matches) const;
  bool areIdsMatching(const Id id1, const Id id2) const;
  /// \brief Add a match between two ids.
  void addMatch(const Id id1, const Id id2);
  const std::vector<Id>& at(const size_t index) const { return id_match_list_.at(index); }
  size_t size() const { return id_match_list_.size(); }
  void clear();
  bool empty() { return id_match_list_.empty(); }
  std::string asString() const;
  // This function should only be used if you intend to modify the returned vector.
  // Otherwise, use .at() to access UniqueIdMatches data.
  std::vector<std::vector<Id> > getIdMatchList() const { return id_match_list_; };
  bool findId(const Id id, Position* position=NULL) const;

 private:
  std::vector<std::vector<Id> > id_match_list_;
}; // class UniqueIdMatches

struct MergeEvent {
  MergeEvent(laser_slam::Time time_ns, Id id_b, Id id_a) :
    timestamp_ns(time_ns),
    id_before(id_b),
    id_after(id_a) {}
  laser_slam::Time timestamp_ns;
  Id id_before;
  Id id_after;
};

bool exportSessionDataToDatabase(const SegmentedCloud& segmented_cloud,
                                 const UniqueIdMatches& id_matches);
bool importSessionDataFromDatabase(SegmentedCloud* segmented_cloud_ptr,
                                   UniqueIdMatches* id_matches_ptr);

bool ensureDirectoryExists(const std::string& directory);
bool ensureDirectoryExistsForFilename(const std::string& filename);

bool exportSegments(const std::string& filename,
                    const SegmentedCloud& segmented_cloud,
                    bool export_all_views = false,
                    bool export_reconstructions = false);
bool exportPositions(const std::string& filename,
                     const SegmentedCloud& segmented_cloud,
                     bool export_all_views = false);
bool exportFeatures(const std::string& filename,
                    const SegmentedCloud& segmented_cloud,
                    bool export_all_views = false);
bool exportSegmentsAndFeatures(const std::string& filename_prefix,
                               const SegmentedCloud& segmented_cloud,
                               bool export_all_views = false);
bool exportSegmentsTimestamps(const std::string& filename,
                              const SegmentedCloud& segmented_cloud,
                              bool export_all_views = false);
bool exportMatches(const std::string& filename, const UniqueIdMatches& matches);
bool exportMergeEvents(const std::string& filename,
                       const std::vector<MergeEvent>& merge_events);
bool importSegments(const std::string& filename,
                    SegmentedCloud* segmented_cloud_ptr);
bool importFeatures(const std::string& filename,
                    SegmentedCloud* segmented_cloud_ptr,
                    const std::string& behavior_when_segment_has_features="abort");
bool importMatches(const std::string& filename, UniqueIdMatches* matches_ptr);

} // namespace database
} // namespace segmatch

#endif // SEGMATCH_DATABASE_HPP_
