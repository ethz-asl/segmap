#ifndef SEGMATCH_SEGMATCH_HPP_
#define SEGMATCH_SEGMATCH_HPP_

#include <cmath>
#include <queue>
#include <string>

#include <laser_slam/common.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "segmatch/common.hpp"
#include "segmatch/descriptors/descriptors.hpp"
#include "segmatch/opencv_random_forest.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmenters/segmenters.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

struct SegMatchParams {
  double segmentation_radius_m;
  bool filter_boundary_segments;
  double boundary_radius_m;
  bool filter_duplicate_segments;
  double centroid_distance_threshold_m;
  laser_slam::Time min_time_between_segment_for_matches_ns;

  DescriptorsParameters descriptors_params;
  SegmenterParameters segmenter_params;
  ClassifierParams classifier_params;
  GeometricConsistencyParams geometric_consistency_params;
};

class SegMatch {
 public:
  explicit SegMatch(const SegMatchParams& params);
  SegMatch();
  ~SegMatch();

  /// \brief Init SegMatch.
  void init(const SegMatchParams& params,
            unsigned int num_tracks = 1u);

  /// \brief Convenience function for setting part of the params.
  void setParams(const SegMatchParams& params);

  /// \brief Process a source cloud.
  void processAndSetAsSourceCloud(const PointICloud& source_cloud,
                                  const laser_slam::Pose& latest_pose,
                                  const unsigned int track_id);

  /// \brief Process a target cloud.
  void processAndSetAsTargetCloud(const PointICloud& target_cloud);

  /// \brief Transfer the source cloud to the target cloud.
  void transferSourceToTarget();

  /// \brief Find matches between the source and the target clouds.
  PairwiseMatches findMatches(PairwiseMatches* matches_after_first_stage = NULL);

  /// \brief Find nearest neighbours between the source and target segments.
  PairwiseMatches findNearestNeighbours() { };

  /// \brief Find the most promising group of consistent matches.
  bool filterMatches(const PairwiseMatches& predicted_matches,
                     PairwiseMatches* filtered_matches_ptr,
                     laser_slam::RelativePose* loop_closure = NULL,
                     std::vector<PointICloudPair>* matched_segment_clouds = NULL);

  void update(const std::vector<laser_slam::Trajectory>& trajectories);

  /// \brief Get the internal representation of the source cloud.
  void getSourceRepresentation(PointICloud* source_representation,
                               const double& distance_to_raise = 0.0) const;

  /// \brief Get the internal representation of the target cloud.
  void getTargetRepresentation(PointICloud* target_representation) const;

  void getTargetSegmentsCentroids(PointICloud* segments_centroids) const;

  void getSourceSegmentsCentroids(PointICloud* segments_centroids) const;

  SegmentedCloud getSourceAsSegmentedCloud() const { return segmented_source_cloud_; };

  SegmentedCloud getTargetAsSegmentedCloud() const { return segmented_target_cloud_; };

  void getPastMatchesRepresentation(PointPairs* past_matches,
                                    PointPairs* invalid_past_matches = NULL) const;

  void getLatestMatch(int64_t* time_a, int64_t* time_b,
                      Eigen::Matrix4f* transform_a_b,
                      std::vector<int64_t>* collector_times) const;

  /// \brief Process a cloud and return the segmented cloud.
  void processCloud(const PointICloud& source_cloud, SegmentedCloud* segmented_cloud,
                    std::vector<double>* timings = NULL);

  /// \brief Get the descriptors dimension.
  unsigned int getDescriptorsDimension() const { return descriptors_->dimension(); };

  /// \brief Train the classifier.
  void trainClassifier(const Eigen::MatrixXd& features, const Eigen::MatrixXd& labels) {
    classifier_->train(features, labels);
  };

  /// \brief Test the classifier.
  void testClassifier(const Eigen::MatrixXd& features, const Eigen::MatrixXd& labels,
                      Eigen::MatrixXd* probabilities = NULL) {
    classifier_->test(features, labels, probabilities);
  };

  /// \brief Save the classifier.
  void saveClassifier(const std::string& filename) { classifier_->save(filename); };

  /// \brief Load a classifier.
  void loadClassifier(const std::string& filename) { classifier_->load(filename); };

  void computeFeaturesDistance(const Eigen::MatrixXd& f1, const Eigen::MatrixXd& f2,
                               Eigen::MatrixXd* f_out) const {
    classifier_->computeFeaturesDistance(f1, f2, f_out);
  };

  void getSegmentationPoses(std::vector<laser_slam::Trajectory>* poses) const {
    CHECK_NOTNULL(poses);
    *poses = segmentation_poses_;
  };

  segmatch::PairwiseMatches  getFilteredMatches() const { return last_filtered_matches_; };
  segmatch::PairwiseMatches  getPredictedMatches() const { return last_predicted_matches_; };

  void getLoopClosures(std::vector<laser_slam::RelativePose>* loop_closures) const;

  void alignTargetMap();

 private:
  void filterBoundarySegmentsOfSourceCloud(const PclPoint& center);

  void filterDuplicateSegmentsOfTargetMap(const SegmentedCloud& cloud_to_be_added);

  laser_slam::Time findTimeOfClosestSegmentationPose(const segmatch::Segment& segment) const;

  SegMatchParams params_;

  std::unique_ptr<Segmenter> segmenter_;
  std::unique_ptr<Descriptors> descriptors_;

  //TODO(Renaud or Daniel): modify with base class when needed.
  std::unique_ptr<OpenCvRandomForest> classifier_;

  SegmentedCloud segmented_source_cloud_;
  SegmentedCloud segmented_target_cloud_;
  std::vector<SegmentedCloud> target_queue_;

  // Contains the poses where segmentation and matching was performed.
  std::vector<laser_slam::Trajectory> segmentation_poses_;

  PairwiseMatches last_filtered_matches_;
  PairwiseMatches last_predicted_matches_;

  std::vector<laser_slam::RelativePose> loop_closures_;

  Eigen::Matrix4f last_transformation_;

  // Filtering parameters.
  static constexpr double kCylinderHeight_m = 40;
  static constexpr unsigned int kMaxNumberOfCloudToTransfer = 1u;

  static constexpr laser_slam::Time kMaxTimeDiffBetweenSegmentAndPose_ns = 20000000000u;

}; // class SegMatch

} // namespace segmatch

#endif // SEGMATCH_SEGMATCH_HPP_
