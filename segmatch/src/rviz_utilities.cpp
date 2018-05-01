#include "segmatch/rviz_utilities.hpp"

namespace segmatch {

PointICloud RVizUtilities::segmentedCloudtoPointICloud(
    const SegmentedCloud& segmented_cloud, bool use_point_cloud_to_publish,
    bool use_reconstruction) {
  // TODO: Once we start saving segment descriptors instead of segments, do
  // this already while adding segments to the segmented cloud.


  // Reserve space for the point cloud
  PointICloud cloud;
  size_t cloud_size = 0;
  for (const auto& segment : segmented_cloud) {
    if (use_point_cloud_to_publish) {
      cloud_size += segment.second.getLastView().point_cloud_to_publish.size();
    } else if (use_reconstruction) {
      cloud_size += segment.second.getLastView().reconstruction.size();
    } else {
      cloud_size += segment.second.getLastView().point_cloud.size();
    }
  }
  cloud.reserve(cloud_size);

  // Copy and points and assign segment colors.
  for (const auto& segment : segmented_cloud) {
    float segment_color = getSegmentColorAsIntensity(segment.first);
    if (use_point_cloud_to_publish) {
      for (const auto& point : segment.second.getLastView().point_cloud_to_publish) {
        cloud.push_back(PointI(segment_color));
        cloud.back().getArray3fMap() = point.getArray3fMap();
      }
    } else if (use_reconstruction) {
      for (const auto& point : segment.second.getLastView().reconstruction) {
        cloud.push_back(PointI(segment_color));
        cloud.back().getArray3fMap() = point.getArray3fMap();
      }
    } else {
      for (const auto& point : segment.second.getLastView().point_cloud) {
        cloud.push_back(PointI(segment_color));
        cloud.back().getArray3fMap() = point.getArray3fMap();
      }
    }

  }

  cloud.width = 1;
  cloud.height = cloud.points.size();
  return cloud;
}

PointICloud RVizUtilities::segmentedCloudSemanticstoPointICloud(
    const SegmentedCloud& segmented_cloud, bool use_reconstruction,
    bool get_compressed) {
  // TODO: Once we start saving segment descriptors instead of segments, do
  // this already while adding segments to the segmented cloud.

  // Reserve space for the point cloud
  PointICloud cloud;
  size_t cloud_size = 0;
  for (const auto& segment : segmented_cloud) {
    if (use_reconstruction) {
      if (get_compressed) {
        cloud_size += segment.second.getLastView().reconstruction_compressed.size();
      } else {
        cloud_size += segment.second.getLastView().reconstruction.size();
      }
    } else {
      cloud_size += segment.second.getLastView().point_cloud.size();
    }
  }
  cloud.reserve(cloud_size);

  // Copy and points and assign segment colors.
  for (const auto& segment : segmented_cloud) {
      PointI segment_color;
    if (segment.second.getLastView().semantic == 0) { // Others
        segment_color = 120;
    } else if (segment.second.getLastView().semantic == 1) { // Cars
        segment_color = 0;
    } else { // Buildings
        segment_color = 220;
    }
    if (use_reconstruction) {
      if (get_compressed) {
        for (const auto& point : segment.second.getLastView().reconstruction_compressed) {
          cloud.push_back(segment_color);
          cloud.back().getArray3fMap() = point.getArray3fMap();
        }
      } else {
        for (const auto& point : segment.second.getLastView().reconstruction) {
          cloud.push_back(segment_color);
          cloud.back().getArray3fMap() = point.getArray3fMap();
        }
      }
    } else {
      for (const auto& point : segment.second.getLastView().point_cloud) {
        cloud.push_back(segment_color);
        cloud.back().getArray3fMap() = point.getArray3fMap();
      }
    }
  }

  cloud.width = 1;
  cloud.height = cloud.points.size();
  return cloud;
}

float RVizUtilities::getSegmentColorAsIntensity(const Id segment_id) {
  // Prevent bugs in case the typedef of Id is changed.
  static_assert(std::is_same<int64_t, Id>::value,
                "The hashing function of the segment Id must be modified if "
                "the type of Id is changed.");

  // Not-so-fancy permutation, based on: http://stackoverflow.com/q/538738
  Id hashed_id = ((((segment_id ^ 0xf7f7f7f7f7f7f7f7) * 0x8364abf78364abf7)
      ^ 0xf00bf00bf00bf00b) * 0xf81bc437f81bc437);

  // Restrict the range of possible intensities to [0 4096] (default in RViz).
  return static_cast<float>(std::abs(hashed_id) % 4096);
}

} // namespace segmatch
