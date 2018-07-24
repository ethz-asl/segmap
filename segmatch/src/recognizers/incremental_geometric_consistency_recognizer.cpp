#include "segmatch/recognizers/incremental_geometric_consistency_recognizer.hpp"

#include <limits>
#include <vector>

#include <glog/logging.h>
#include <laser_slam/benchmarker.hpp>

#include "segmatch/common.hpp"
#include "segmatch/recognizers/graph_utilities.hpp"
#include "segmatch/recognizers/matches_partitioner.hpp"

namespace segmatch {

constexpr size_t IncrementalGeometricConsistencyRecognizer::kNoMatchIndex_;
constexpr size_t IncrementalGeometricConsistencyRecognizer::kNoCacheSlotIndex_;

IncrementalGeometricConsistencyRecognizer::IncrementalGeometricConsistencyRecognizer(
    const GeometricConsistencyParams& params, const float max_model_radius) noexcept
  : GraphBasedGeometricConsistencyRecognizer(params)
  , max_consistency_distance_(max_model_radius * 2.0 + params.resolution)
  , max_consistency_distance_for_caching_(
      params.max_consistency_distance_for_caching + params.resolution)
  , half_max_consistency_distance_for_caching_(params.max_consistency_distance_for_caching * 0.5f) {
}

inline void IncrementalGeometricConsistencyRecognizer::processCachedMatches(
    const PairwiseMatches& predicted_matches,
    const std::vector<MatchLocations>& cached_matches_locations,
    const std::vector<size_t>& cache_slot_index_to_match_index,
    std::unordered_map<IdPair, size_t, IdPairHash>& new_cache_slot_indices,
    ConsistencyGraph& consistency_graph) {
  BENCHMARK_BLOCK("SM.Worker.Recognition.BuildConsistencyGraph.CachedMatches");

  // Recompute consistency information of cached elements where necessary.
  size_t num_consistency_tests = 0u;
  for (const auto& cached_match_locations : cached_matches_locations) {
    const PairwiseMatch& match = predicted_matches[cached_match_locations.match_index];
    MatchCacheSlot& match_cache = matches_cache_[cached_match_locations.cache_slot_index];
    new_cache_slot_indices.emplace(match.ids_, cached_match_locations.cache_slot_index);

    // For each cached element, get rid of any reference to matches that do not exist anymore and
    // add consistent pairs to the consistency graph.
    std::vector<size_t> new_candidate_consistent_matches;
    new_candidate_consistent_matches.reserve(match_cache.candidate_consistent_matches.size());
    for (const size_t candidate_cache_slot_index : match_cache.candidate_consistent_matches) {
      const size_t match_2_index = cache_slot_index_to_match_index[candidate_cache_slot_index];
      if (match_2_index != kNoMatchIndex_) {
        const PairwiseMatch& match_2 = predicted_matches[match_2_index];
        float consistency_distance = computeConsistencyDistance(match, match_2,
                                                                max_consistency_distance_);
        ++num_consistency_tests;

        // If the matches are close enough, cache them as candidate consistent matches.
        if (consistency_distance <= max_consistency_distance_) {
          new_candidate_consistent_matches.emplace_back(candidate_cache_slot_index);

          // If the matches are consistent, add and edge to the consistency graph
          if (consistency_distance <= params_.resolution)
            boost::add_edge(match_2_index, cached_match_locations.match_index,
                            consistency_graph);
        }
      }
    }
    match_cache.candidate_consistent_matches = std::move(new_candidate_consistent_matches);
  }
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.TestedCachedPairs",
                         num_consistency_tests);
}

inline void IncrementalGeometricConsistencyRecognizer::processNewMatches(
    const PairwiseMatches& predicted_matches,
    const std::vector<size_t>& free_cache_slot_indices,
    std::vector<size_t>& match_index_to_cache_slot_index,
    std::unordered_map<IdPair, size_t, IdPairHash>& new_cache_slot_indices,
    ConsistencyGraph& consistency_graph) {
  BENCHMARK_BLOCK("SM.Worker.Recognition.BuildConsistencyGraph.NewMatches");

  // Partition the matches in a grid by the position of the scene points. The size of the
  // partitions is greater or equal the size of the model. This way we can safely assume that, if
  // the model is actually present in the scene, all matches will be contained in a 2x2 group of
  // adjacent partitions.
  BENCHMARK_START("SM.Worker.Recognition.BuildConsistencyGraph.Partitioning");
  MatchesGridPartitioning<PartitionData> partitioning =
      MatchesPartitioner::computeGridPartitioning<PartitionData>(predicted_matches,
                                                                 max_consistency_distance_);
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.NumPartitions",
                         partitioning.getHeight() * partitioning.getWidth());
  BENCHMARK_STOP("SM.Worker.Recognition.BuildConsistencyGraph.Partitioning");

  // Find all possible consistency within a partition and within neighbor partitions.
  size_t num_consistency_tests = 0u;
  size_t next_slot_index_position = 0u;
  for (size_t i = 0; i < partitioning.getHeight(); ++i) {
    for (size_t j = 0; j < partitioning.getWidth(); ++j) {
      for (const auto match_index : partitioning(i, j).match_indices) {
        // Only process new matches.
        if (match_index_to_cache_slot_index[match_index] != kNoCacheSlotIndex_) continue;
        const PairwiseMatch& match = predicted_matches[match_index];

        // Get a free cache slot and insert the match.
        const size_t cache_slot_index = free_cache_slot_indices[next_slot_index_position];
        ++next_slot_index_position;
        MatchCacheSlot& match_cache = matches_cache_[cache_slot_index];
        match_cache.candidate_consistent_matches.clear();
        match_cache.candidate_consistent_matches.reserve(matches_cache_.size() - 1u);
        match_cache.centroids_at_caching = match.centroids_;
        new_cache_slot_indices.emplace(match.ids_, cache_slot_index);

        // Test consistencies between the current match and the cached matches in the neighbor
        // partitions.
        for (size_t k = static_cast<size_t>(std::max(0, static_cast<int>(i) - 1));
             k <= std::min(partitioning.getHeight() - 1u, i + 1u); ++k) {
          for (size_t l = static_cast<size_t>(std::max(0, static_cast<int>(j) - 1));
               l <= std::min(partitioning.getWidth() - 1u, j + 1u); ++l) {
            for (const auto match_2_index : partitioning(k, l).match_indices) {
              // Only compare to matches already present in the cache
              if (match_index_to_cache_slot_index[match_2_index] != kNoCacheSlotIndex_) {
                const PairwiseMatch& match_2 = predicted_matches[match_2_index];
                float consistency_distance = computeConsistencyDistance(match, match_2,
                                                                        max_consistency_distance_);
                ++num_consistency_tests;

                // If the matches are close enough, cache them as candidate consistent matches.
                if (consistency_distance <= max_consistency_distance_for_caching_) {
                  match_cache.candidate_consistent_matches.emplace_back(
                      match_index_to_cache_slot_index[match_2_index]);
                  // If the matches are consistent, add an edge to the consistency graph.
                  if (consistency_distance <= params_.resolution)
                    boost::add_edge(match_index, match_2_index, consistency_graph);
                }
              }
            }
          }
        }

        match_index_to_cache_slot_index[match_index] = cache_slot_index;
      }
    }
  }
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.TestedNewPairs",
                           num_consistency_tests);
}

bool IncrementalGeometricConsistencyRecognizer::mustRemoveFromCache(
    const PairwiseMatch& match, const size_t cache_slot_index) {
  const MatchCacheSlot& match_cache = matches_cache_[cache_slot_index];
  pcl::Vector3fMapConst model_centroid = match.centroids_.first.getVector3fMap();
  pcl::Vector3fMapConst scene_centroid = match.centroids_.second.getVector3fMap();
  pcl::Vector3fMapConst model_centroid_at_caching =
      match_cache.centroids_at_caching.first.getVector3fMap();
  pcl::Vector3fMapConst scene_centroid_at_caching =
      match_cache.centroids_at_caching.second.getVector3fMap();

  // Since checking the change in consistency distance for every match pair would be too expensive,
  // we the responsibility of the check on both matches. If the centroids of a match move by half
  // the maximum distance allowed, then the cached information are invalidated independently of the
  // changes of the other matches.
  const float model_displacement = (model_centroid - model_centroid_at_caching).norm();
  const float scene_displacement = (scene_centroid - scene_centroid_at_caching).norm();
  return model_displacement + scene_displacement >= half_max_consistency_distance_for_caching_;
}

inline IncrementalGeometricConsistencyRecognizer::ConsistencyGraph
IncrementalGeometricConsistencyRecognizer::buildConsistencyGraph(
    const PairwiseMatches& predicted_matches) {
  BENCHMARK_BLOCK("SM.Worker.Recognition.BuildConsistencyGraph");

  // Resize the cache to fit the new matches.
  if (predicted_matches.size() > matches_cache_.size())
    matches_cache_.resize(predicted_matches.size());
  std::vector<size_t> cache_slot_index_to_match_index(matches_cache_.size(), kNoMatchIndex_);

  // Identify which matches have cached information.
  size_t invalidated_cached_matches = 0u;
  std::vector<MatchLocations> cached_matches_locations;
  cached_matches_locations.reserve(predicted_matches.size());
  std::vector<size_t> match_index_to_cache_slot_index(predicted_matches.size(),
                                                      kNoCacheSlotIndex_);
  for (size_t i = 0u; i < predicted_matches.size(); ++i) {
    const auto cached_info_it = cache_slot_indices_.find(predicted_matches[i].ids_);
    if (cached_info_it != cache_slot_indices_.end()) {
      // If a centroid moved by more than the allowed distance, we need to invalidate the cached
      // information and threat the match as new.
      if (mustRemoveFromCache(predicted_matches[i], cached_info_it->second)) {
        ++invalidated_cached_matches;
      } else {
        cached_matches_locations.emplace_back(i, cached_info_it->second);
        cache_slot_index_to_match_index[cached_info_it->second] = i;
        match_index_to_cache_slot_index[i] = cached_info_it->second;
      }
    }
  }
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.InvalidatedMatches",
                         invalidated_cached_matches);
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.BuildConsistencyGraph.CachedMatches",
                         cached_matches_locations.size());

  // Collect indices of the cache slots that are not used anymore.
  std::vector<size_t> free_cache_slot_indices;
  free_cache_slot_indices.reserve(matches_cache_.size());
  for (size_t i = 0u; i < matches_cache_.size(); ++i) {
    if (cache_slot_index_to_match_index[i] == kNoMatchIndex_)
      free_cache_slot_indices.push_back(i);
  }

  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.TotalMatches", predicted_matches.size());
  BENCHMARK_RECORD_VALUE("SM.Worker.Recognition.CachedMatches", cached_matches_locations.size());

  // Build the consistency graph.
  ConsistencyGraph consistency_graph(predicted_matches.size());
  std::unordered_map<IdPair, size_t, IdPairHash> new_cache_slot_indices;
  new_cache_slot_indices.reserve(predicted_matches.size());
  processCachedMatches(predicted_matches, cached_matches_locations,
                       cache_slot_index_to_match_index, new_cache_slot_indices, consistency_graph);
  processNewMatches(predicted_matches, free_cache_slot_indices, match_index_to_cache_slot_index,
                    new_cache_slot_indices, consistency_graph);

  // Use the new mapping between match IDs and cache slots.
  cache_slot_indices_ = std::move(new_cache_slot_indices);
  return consistency_graph;
}

inline float IncrementalGeometricConsistencyRecognizer::computeConsistencyDistance(
    const PairwiseMatch& first_match, const PairwiseMatch& second_match,
    const float max_target_distance) const {
  // Get the centroids of the matched segments.
  pcl::Vector3fMapConst model_point_1 = first_match.centroids_.first.getVector3fMap();
  pcl::Vector3fMapConst scene_point_1 = first_match.centroids_.second.getVector3fMap();
  pcl::Vector3fMapConst model_point_2 = second_match.centroids_.first.getVector3fMap();
  pcl::Vector3fMapConst scene_point_2 = second_match.centroids_.second.getVector3fMap();

  // If the keypoints are so far away in the scene so that they can not fit in the model together,
  // the matches will not be consistent even if the centroids in the model move.
  const float scene_distance = (scene_point_1 - scene_point_2).norm();
  if (scene_distance > max_target_distance)
    return std::numeric_limits<float>::max();

  // Return the difference between the distances.
  const float model_distance = (model_point_1 - model_point_2).norm();
  return fabs(scene_distance - model_distance);
}

} // namespace segmatch
