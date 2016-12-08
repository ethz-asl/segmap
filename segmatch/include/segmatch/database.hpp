#ifndef SEGMATCH_DATABASE_HPP_
#define SEGMATCH_DATABASE_HPP_

#include <string>

#include <segmatch/segmented_cloud.hpp>

namespace segmatch {

// TODO: Replace IdMatches?
/// \brief A structure for storing a position within a 2D table.
struct Position {
  size_t row = 0;
  size_t col = 0;
};

class IdMatches {
 public:
  IdMatches() {}
  explicit IdMatches(std::vector<std::vector<Id> > vector) { id_match_list_ = vector; }
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
  // Otherwise, use .at() to access IdMatches data.
  std::vector<std::vector<Id> > getIdMatchList() const { return id_match_list_; };
  bool findId(const Id id, Position* position=NULL) const;

 private:
  std::vector<std::vector<Id> > id_match_list_;
}; // class IdMatches

} // namespace segmatch

bool export_session_data_to_database(const segmatch::SegmentedCloud& segmented_cloud,
                                     const segmatch::IdMatches& id_matches);
bool import_session_data_from_database(segmatch::SegmentedCloud* segmented_cloud_ptr,
                                       segmatch::IdMatches* id_matches_ptr);

bool ensure_directory_exists(const std::string& directory);
bool ensure_directory_exists_for_filename(const std::string& filename);

bool export_segments(const std::string& filename,
                     const segmatch::SegmentedCloud& segmented_cloud);
bool export_features(const std::string& filename,
                     const segmatch::SegmentedCloud& segmented_cloud);
bool export_features_and_centroids(const std::string& filename,
                                   const segmatch::SegmentedCloud& segmented_cloud);
bool export_matches(const std::string& filename, const segmatch::IdMatches& matches);

bool import_segments(const std::string& filename,
                     segmatch::SegmentedCloud* segmented_cloud_ptr);
bool import_features(const std::string& filename,
                     segmatch::SegmentedCloud* segmented_cloud_ptr,
                     const std::string& behavior_when_segment_has_features="abort");
bool import_matches(const std::string& filename, segmatch::IdMatches* matches_ptr);

#endif // SEGMATCH_DATABASE_HPP_
