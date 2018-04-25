#include "segmatch/database.hpp"

#include <fstream>

#include <boost/filesystem.hpp>
#include <glog/logging.h>

#include "segmatch/utilities.hpp"

namespace segmatch {
namespace database {

bool UniqueIdMatches::findMatches(const Id id, std::vector<Id>* matches) const {
  CHECK_NOTNULL(matches)->clear();
  Position position;
  if (findId(id, &position)) {
    // Copy the matches and remove oneself.
    *matches = id_match_list_.at(position.row);
    matches->erase(matches->begin() + position.col);
    return true;
  }
  return false;
}

bool UniqueIdMatches::areIdsMatching(const Id id1, const Id id2) const {
  Position position1;
  Position position2;
  if (findId(id1, &position1) && findId(id2, &position2)) {
    return position1.row == position2.row;
  }
  return false;
}

void UniqueIdMatches::addMatch(const Id id1, const Id id2) {
  CHECK_NE(id1, id2) << "No point in adding match between identical ids.";
  Position position1;
  Position position2;
  if (findId(id1, &position1)) {
    if (findId(id2, &position2)) {
      // Found both.
      if (position1.row != position2.row) {
        // Each id is already in separate groups -> concatenate both groups into one.
        std::vector<Id> matches2_copy = id_match_list_.at(position2.row);
        id_match_list_.erase(id_match_list_.begin() + position2.row);
        CHECK(findId(id1, &position1));
        id_match_list_.at(position1.row).insert(id_match_list_.at(position1.row).begin(),
                                                matches2_copy.begin(), matches2_copy.end());
        return;
      } else {
        // Match already exists.
        return;
      }
    } else {
      // Found id1 but not id2 -> add id1 to id2's matches' group.
      id_match_list_.at(position1.row).push_back(id2);
      return;
    }
  } else if (findId(id2, &position2)) {
    // Found id2 but not id1 -> add id1 to id2's matches' group.
    id_match_list_.at(position2.row).push_back(id1);
    return;
  } else {
    // Found neither.
    std::vector<Id> match_to_add;
    match_to_add.push_back(id1);
    match_to_add.push_back(id2);
    id_match_list_.push_back(match_to_add);
    return;
  }
}

void UniqueIdMatches::clear() {
  id_match_list_.clear();
}

std::string UniqueIdMatches::asString() const {
  std::stringstream result;
  for (size_t i = 0u; i < id_match_list_.size(); ++i) {
    for (size_t j = 0u; j < id_match_list_.at(i).size(); ++j) {
      result << id_match_list_.at(i).at(j) << " ";
    }
    result << std::endl;
  }
  return result.str();
}

bool UniqueIdMatches::findId(const Id id, Position* position_ptr) const {
  for (size_t i = 0u; i < id_match_list_.size(); ++i) {
    for (size_t j = 0u; j < id_match_list_.at(i).size(); ++j) {
      if (id == id_match_list_.at(i).at(j)) {
        // If desired, pass the found id's position.
        if (position_ptr != NULL) {
          position_ptr->row = i;
          position_ptr->col = j;
        }
        return true;
      }
    }
  }
  return false;
}

const std::string kDatabaseDirectory = "/tmp/segmatch/";
const std::string kSegmentsFilename = "segments_database.csv";
const std::string kFeaturesFilename = "features_database.csv";
const std::string kMatchesFilename = "matches_database.csv";

bool exportSessionDataToDatabase(const SegmentedCloud& segmented_cloud,
                                 const UniqueIdMatches& id_matches) {
  return exportSegments(kDatabaseDirectory + kSegmentsFilename, segmented_cloud) &&
      exportFeatures(kDatabaseDirectory + kFeaturesFilename, segmented_cloud) &&
      exportMatches(kDatabaseDirectory + kMatchesFilename, id_matches);
}

bool importSessionDataFromDatabase(SegmentedCloud* segmented_cloud_ptr,
                                   UniqueIdMatches* id_matches_ptr) {
  return importSegments(kDatabaseDirectory + kSegmentsFilename, segmented_cloud_ptr) &&
      importFeatures(kDatabaseDirectory + kFeaturesFilename, segmented_cloud_ptr) &&
      importMatches(kDatabaseDirectory + kMatchesFilename, id_matches_ptr);
}

bool ensureDirectoryExists(const std::string& directory) {
  CHECK(directory.size() != 0u) << "Directory should not be an empty string.";
  if (directory[0u] == '/') {
    boost::filesystem::path path(directory);
    if(boost::filesystem::exists(path)) {
      return true;
    }
    if(boost::filesystem::create_directory(path)) {
      LOG(WARNING) << "Directory Created: " << directory;
      return true;
    }
  } else {
    LOG(ERROR) << "Directory '" << directory << "' starts with invalid character: '" <<
        directory[0u] << "'"; 
  }
  return false;
}

bool ensureDirectoryExistsForFilename(const std::string& filename) {
  size_t pos = filename.rfind("/");
  if (pos != std::string::npos) {
    std::string directory = filename.substr(0u, pos);
    return ensureDirectoryExists(directory);
  } else {
    LOG(ERROR) << "Filename '" << filename << "' does not specify a directory.";
    return false;
  }
}

bool exportSegments(const std::string& filename, const SegmentedCloud& segmented_cloud,
                    const bool export_all_views, bool export_reconstructions) {
  ensureDirectoryExistsForFilename(filename);
  std::ofstream output_file;
  output_file.open(filename, std::ofstream::out | std::ofstream::trunc);
  if (output_file.is_open()) {
    for (std::unordered_map<Id, Segment>::const_iterator it = segmented_cloud.begin();
        it != segmented_cloud.end(); ++it) {
      Segment segment = it->second;
      if (export_all_views) {
        for (size_t i = 0u; i < segment.views.size(); ++i) {
          PointCloud point_cloud;
          if (export_reconstructions) {
            point_cloud = segment.views[i].reconstruction;
          } else {
            point_cloud = segment.views[i].point_cloud;
          }
          for (const auto& point : point_cloud) {
            output_file << segment.segment_id << " ";
            output_file << i << " "; // Index of the view.
            output_file << point.x << " ";
            output_file << point.y << " ";
            output_file << point.z;
            output_file << std::endl;
          }
        }
      } else {
        PointCloud point_cloud;
        if (export_reconstructions) {
          point_cloud = segment.getLastView().reconstruction;
        } else {
          point_cloud = segment.getLastView().point_cloud;
        }
        for (const auto& point : point_cloud) {
          output_file << segment.segment_id << " ";
          output_file << point.x << " ";
          output_file << point.y << " ";
          output_file << point.z;
          output_file << std::endl;
        }
      }
    }
    output_file.close();
    LOG(INFO) << segmented_cloud.getNumberOfValidSegments() << " segments written to " << filename;
    return true;
  } else {
    LOG(ERROR) << "Could not open file " << filename << " for writing segments.";
    return false;
  }
}

bool exportPositions(const std::string& filename, const SegmentedCloud& segmented_cloud,
                     const bool export_all_views){
  ensureDirectoryExistsForFilename(filename);
  std::ofstream output_file;
  output_file.open(filename, std::ofstream::out | std::ofstream::trunc);
  if (output_file.is_open()) {
    for (std::unordered_map<Id, Segment>::const_iterator it = segmented_cloud.begin();
        it != segmented_cloud.end(); ++it) {
      Segment segment = it->second;
      if (export_all_views) {
        for (size_t i = 0u; i < segment.views.size(); ++i) {
          SE3::Position pos = segment.views[i].T_w_linkpose.getPosition();
          output_file << segment.segment_id << " ";
          output_file << i << " "; // Index of the view.
          output_file << pos[0] << " ";
          output_file << pos[1] << " ";
          output_file << pos[2] << " ";
          output_file << std::endl;
        }
      } else {
        SE3::Position pos = segment.getLastView().T_w_linkpose.getPosition();
        output_file << segment.segment_id << " ";
        output_file << pos[0] << " ";
        output_file << pos[1] << " ";
        output_file << pos[2] << " ";
        output_file << std::endl;
      }
    }
    output_file.close();
    LOG(INFO) << segmented_cloud.getNumberOfValidSegments() << " segments written to " << filename;
    return true;
  } else {
    LOG(ERROR) << "Could not open file " << filename << " for writing segment positions.";
    return false;
  }	
}

bool exportFeatures(const std::string& filename, const SegmentedCloud& segmented_cloud,
                    const bool export_all_views) {
  ensureDirectoryExistsForFilename(filename);
  std::ofstream output_file;
  output_file.open(filename, std::ofstream::out | std::ofstream::trunc);
  if (output_file.is_open()) {
    for (std::unordered_map<Id, Segment>::const_iterator it = segmented_cloud.begin();
        it != segmented_cloud.end(); ++it) {
      Segment segment = it->second;
      if (export_all_views) {
        for (size_t i = 0u; i < segment.views.size(); ++i) {
          output_file << segment.segment_id << " ";
          output_file << i << " "; // Index of the view.
          std::vector<FeatureValueType> values = segment.views[i].features.asVectorOfValues();
          std::vector<std::string> names = segment.views[i].features.asVectorOfNames();
          for (size_t j = 0u; j < values.size(); ++j) {
            output_file << names.at(j) << " ";
            output_file << values.at(j) << " ";
          }
          output_file << std::endl;
        }
      } else {
        output_file << segment.segment_id << " ";
        std::vector<FeatureValueType> values = segment.getLastView().features.asVectorOfValues();
        std::vector<std::string> names = segment.getLastView().features.asVectorOfNames();
        for (size_t j = 0u; j < values.size(); ++j) {
          output_file << names.at(j) << " ";
          output_file << values.at(j) << " ";
        }
        output_file << std::endl;
      }
    }
    output_file.close();
    LOG(INFO) << "Features written to " << filename;
    return true;
  } else {
    LOG(ERROR) << "Could not open file " << filename << " for writing features.";
    return false;
  }
}

bool exportSegmentsTimestamps(const std::string& filename,
                              const SegmentedCloud& segmented_cloud,
                              const bool export_all_views) {
  ensureDirectoryExistsForFilename(filename);
  std::ofstream output_file;
  output_file.open(filename, std::ofstream::out | std::ofstream::trunc);
  if (output_file.is_open()) {
    for (std::unordered_map<Id, Segment>::const_iterator it = segmented_cloud.begin();
        it != segmented_cloud.end(); ++it) {
      Segment segment = it->second;
      if (export_all_views) {
        for (size_t i = 0u; i < segment.views.size(); ++i) {
          output_file << segment.segment_id << " ";
          output_file << i << " "; // Index of the view.
          output_file << segment.views[i].timestamp_ns << " ";
          output_file << std::endl;
        }
      } else {
        output_file << segment.segment_id << " ";
        output_file << segment.getLastView().timestamp_ns << " ";
        output_file << std::endl;
      }
    }
    output_file.close();
    LOG(INFO) << "Timestamps written to " << filename;
    return true;
  } else {
    LOG(ERROR) << "Could not open file " << filename << " for writing timestamps.";
    return false;
  }
}

bool exportMergeEvents(const std::string& filename, const std::vector<MergeEvent>& merge_events) {
  ensureDirectoryExistsForFilename(filename);
  std::ofstream output_file;
  output_file.open(filename, std::ofstream::out | std::ofstream::trunc);
  if (output_file.is_open()) {
    for (const auto& merge_event : merge_events) {
      output_file << merge_event.timestamp_ns << " ";
      output_file << merge_event.id_before << " ";
      output_file << merge_event.id_after << " ";
      output_file << std::endl;
    }
    output_file.close();
    LOG(INFO) << "Merge events written to " << filename;
    return true;
  } else {
    LOG(ERROR) << "Could not open file " << filename << " for writing merge events.";
    return false;
  }
}

bool exportSegmentsAndFeatures(const std::string& filename_prefix,
                               const SegmentedCloud& segmented_cloud,
                               const bool export_all_views) {
  exportSegments(filename_prefix + "_segments.csv", segmented_cloud, export_all_views);
  exportFeatures(filename_prefix + "_features.csv", segmented_cloud, export_all_views);
  exportSegmentsTimestamps(filename_prefix + "_timestamps.csv", segmented_cloud, export_all_views);
}

bool exportMatches(const std::string& filename, const UniqueIdMatches& matches) {
  ensureDirectoryExistsForFilename(filename);
  std::ofstream output_file;
  output_file.open(filename, std::ofstream::out | std::ofstream::trunc);
  if (output_file.is_open()) {
    for (size_t i = 0u; i < matches.size(); i++) {
      for (size_t j = 0u; j < matches.at(i).size(); ++j) {
        output_file << matches.at(i).at(j) << " ";
      }
      output_file << std::endl;
    }
    output_file.close();
    LOG(INFO) << "Matches written to " << filename;
    return true;
  } else {
    LOG(ERROR) << "Could not open file " << filename << " for writing matches.";
    return false;
  }
}

bool importSegments(const std::string& filename, SegmentedCloud* segmented_cloud_ptr) {
  CHECK_NOTNULL(segmented_cloud_ptr);
  std::ifstream input_file;
  input_file.open(filename);
  size_t segments_count = 0u;
  if (input_file.good()) {
    // Get the current line.
    std::string line;
    Segment segment;
    segment.views.emplace_back(SegmentView());
    while(getline(input_file, line)) {
      std::istringstream line_as_stream(line);
      // Read first number as segment id.
      Id line_id;
      line_as_stream >> line_id;

      // If id has changed, save segment and create new one.
      if (segment.segment_id != kNoId && line_id != segment.segment_id && !segment.empty()) {
        // Ensure that ids remain unique.
        if (segmented_cloud_ptr->findValidSegmentPtrById(segment.segment_id, NULL)) {
          LOG(WARNING) << "Did not import segment of id " << segment.segment_id <<
              ". A segment with that id already exists.";
        } else {
          segmented_cloud_ptr->addValidSegment(segment);
          ++segments_count;
        }
        segment.clear();
        segment.views.emplace_back(SegmentView());
      }
      segment.segment_id = line_id;
      PclPoint point;
      line_as_stream >> point.x;
      line_as_stream >> point.y;
      line_as_stream >> point.z;
      segment.getLastView().point_cloud.push_back(point);
    }
    // After the loop: Store the last segment.
    if (segment.hasValidId()) {
      if (segmented_cloud_ptr->findValidSegmentPtrById(segment.segment_id, NULL)) {
        LOG(WARNING) << "Did not import segment of id " << segment.segment_id <<
            ". A segment with that id already exists.";
      } else {
        segmented_cloud_ptr->addValidSegment(segment);
        ++segments_count;
      }
    }
    input_file.close();
    LOG(INFO) << "Imported " << segments_count << " segments from file " << filename;
    return true;
  } else {
    LOG(ERROR) << "Could not open file " << filename << " for importing segments.";
    return false;
  }
}

bool importFeatures(const std::string& filename, SegmentedCloud* segmented_cloud_ptr,
                    const std::string& behavior_when_segment_has_features) {
  CHECK_NOTNULL(segmented_cloud_ptr);
  std::ifstream input_file;
  input_file.open(filename);
  size_t segments_count = 0u;
  if (input_file.good()) {
    // Get the current line.
    std::string line;
    while(getline(input_file, line)) {
      std::istringstream line_as_stream(line);
      // Read first number as segment id.
      Id segment_id;
      CHECK(line_as_stream >> segment_id) << "Could not read Id.";
      Segment* segment_ptr;
      if (!segmented_cloud_ptr->findValidSegmentPtrById(segment_id, &segment_ptr)) {
        LOG(ERROR) << "Could not find segment of id " << segment_id <<
            " when importing features for that id.";
      } else {
        // Read features.
        Features features;
        Feature feature;
        std::string name;
        while(line_as_stream >> name) {
          FeatureValueType value;
          line_as_stream >> value;
          feature.push_back(FeatureValue(name, value));
        }
        features.push_back(feature);
        // Check wether segment already has features.
        if (!segment_ptr->getLastView().features.empty()) {
          if (behavior_when_segment_has_features == "concatenate") {
            segment_ptr->getLastView().features += features;
          } else if (behavior_when_segment_has_features == "replace") {
            segment_ptr->getLastView().features = features;
          }else /* behavior_when_segment_has_features == "abort" */ {
            LOG(FATAL) << "Segment " << segment_id <<
                " into which features are being imported already has features, " <<
                "and tolerance has not been set. Aborting.";
          }
        } else {
          segment_ptr->getLastView().features = features;
        }
        ++segments_count;
      }
    }
    input_file.close();
    LOG(INFO) << "Imported features for " << segments_count << " segments from file " << filename;
    return true;
  } else {
    LOG(ERROR) << "Could not open file " << filename << " for importing features.";
    return false;
  }
}

bool importMatches(const std::string& filename, UniqueIdMatches* matches_ptr) {
  CHECK_NOTNULL(matches_ptr);
  if (!matches_ptr->empty()) {
    LOG(ERROR) << "Should not import matches into non-empty UniqueIdMatches object.";
    return false;
  }
  std::ifstream input_file;
  input_file.open(filename);
  size_t matches_count = 0u;
  if (input_file.good()) {
    // Get the current line.
    std::string line;
    std::vector<std::vector<Id> > matches_vector;
    while(getline(input_file, line)) {
      std::istringstream line_as_stream(line);
      Id id;
      std::vector<Id> match_group;
      while(line_as_stream >> id) {
        match_group.push_back(id);
      }
      matches_vector.push_back(match_group);
      ++matches_count;
    }
    *matches_ptr = UniqueIdMatches(matches_vector);
    input_file.close();
    LOG(INFO) << "Imported " << matches_count << " matches from file " << filename;
    return true;
  } else {
    LOG(ERROR) << "Could not open file " << filename << " for importing matches.";
    return false;
  }
}

} // namespace database
} // namespace segmatch
