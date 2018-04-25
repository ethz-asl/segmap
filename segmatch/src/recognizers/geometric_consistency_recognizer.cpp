#include "segmatch/recognizers/geometric_consistency_recognizer.hpp"

#include <algorithm>
#include <vector>

#include <glog/logging.h>
#include <laser_slam/benchmarker.hpp>
#include <pcl/correspondence.h>
#include <pcl/recognition/cg/correspondence_grouping.h>
#include <pcl/recognition/cg/geometric_consistency.h>

#include "segmatch/common.hpp"

namespace segmatch {

void GeometricConsistencyRecognizer::recognize(const PairwiseMatches& predicted_matches) {
  // Clear the current candidates and check if we got matches.
  candidate_transfomations_.clear();
  candidate_matches_.clear();
  if (predicted_matches.empty()) return;

  // Build point clouds out of the centroids for geometric consistency grouping.
  pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
  PointCloudPtr first_cloud(new PointCloud());
  PointCloudPtr second_cloud(new PointCloud());

  // Create clouds for geometric consistency.
  BENCHMARK_START("SM.Worker.Recognition.SetupCorrespondences");
  for (size_t i = 0u; i < predicted_matches.size(); ++i) {
    // First centroid.
    PclPoint first_centroid = predicted_matches.at(i).getCentroids().first;
    first_cloud->push_back(first_centroid);
    // Second centroid.
    PclPoint second_centroid = predicted_matches.at(i).getCentroids().second;
    second_cloud->push_back(second_centroid);
    float squared_distance = 1.0 - predicted_matches.at(i).confidence_;
    correspondences->push_back(pcl::Correspondence(i, i, squared_distance));
  }
  BENCHMARK_STOP("SM.Worker.Recognition.SetupCorrespondences");

  // Perform geometric consistency grouping.
  BENCHMARK_START("SM.Worker.Recognition.GCG");
  RotationsTranslations correspondence_transformations;

  typedef std::vector<pcl::Correspondences> Correspondences;
  Correspondences clustered_corrs;
  pcl::GeometricConsistencyGrouping<PclPoint, PclPoint> geometric_consistency_grouping;
  geometric_consistency_grouping.setGCSize(params_.resolution);
  // SegMatch uses the min_cluster_size parameter as the minimum (inclusive) size of a successful
  // recognition, while the PCL uses the parameter as an exclusive lower bound.
  geometric_consistency_grouping.setGCThreshold(params_.min_cluster_size - 1);
  geometric_consistency_grouping.setInputCloud(first_cloud);
  geometric_consistency_grouping.setSceneCloud(second_cloud);
  geometric_consistency_grouping.setModelSceneCorrespondences(correspondences);
  geometric_consistency_grouping.recognize(correspondence_transformations, clustered_corrs);
  BENCHMARK_STOP("SM.Worker.Recognition.GCG");

  if (!clustered_corrs.empty()) {
    BENCHMARK_BLOCK("SM.Worker.Recognition.SortCandidateRecognitions");
    CHECK_EQ(correspondence_transformations.size(), clustered_corrs.size());

    // Sort cluster in decreasing size order
    std::vector<size_t> ordered_indices(clustered_corrs.size());
    std::iota(ordered_indices.begin(), ordered_indices.end(), 0u);
    std::sort(ordered_indices.begin(), ordered_indices.end(), [&](const size_t i, const size_t j) {
      return clustered_corrs[i].size() > clustered_corrs[j].size();
    });

    candidate_transfomations_.resize(correspondence_transformations.size());
    std::transform(ordered_indices.begin(), ordered_indices.end(),
                   candidate_transfomations_.begin(), [&](const size_t i) {
                     return correspondence_transformations[i];
                   });

    for (const auto& cluster : clustered_corrs) {
      // Catch the cases when PCL returns clusters smaller than the minimum cluster size.
      if (cluster.size() >= params_.min_cluster_size) {
        candidate_matches_.emplace_back();
        // Create pairwise matches
        for (size_t i = 0u; i < cluster.size(); ++i) {
          // TODO: This assumes the matches from which the cloud was created
          //       are indexed in the same way as the cloud.
          //       (i.e. match[i] -> first_cloud[i] with second_cloud[i])
          //       Otherwise, this check will fail.
          CHECK(cluster[i].index_query == cluster[i].index_match);
          candidate_matches_.back().push_back(predicted_matches.at(cluster[i].index_query));
        }
      }
    }
  }
}

} // namespace segmatch
