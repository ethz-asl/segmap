#ifndef SEGMATCH_ROS_SEGMATCH_WORKER_HPP_
#define SEGMATCH_ROS_SEGMATCH_WORKER_HPP_

#include <utility>

#include <laser_slam/common.hpp>
#include <segmatch/database.hpp>
#include <segmatch/local_map.hpp>
#include <segmatch/segmatch.hpp>
#include <std_srvs/Empty.h>

#include "segmatch_ros/common.hpp"

namespace segmatch_ros {

class SegMatchWorker {

 public:
  explicit SegMatchWorker();
  ~SegMatchWorker();

  void init(ros::NodeHandle& nh, const SegMatchWorkerParams& params,
            unsigned int num_tracks = 1u);

  /// Process the local map. This performs segmentation, matching and, if
  /// \c loop_closure is provided, loop closure detection.
  /// \param local_map The current local map.
  /// \param latest_pose The latest known pose of the robot.
  /// \param track_id ID of the track being currently processed.
  /// \param loop_closure If specified, pointer to an object where the result
  /// loop closure will be stored.
  /// \returns True if a loop closure was found.
  bool processLocalMap(
      segmatch::SegMatch::LocalMapT& local_map,
      const laser_slam::Pose& latest_pose,
      unsigned int track_id = 0u,
      laser_slam::RelativePose* loop_closure = NULL);

  void update(const laser_slam::Trajectory& trajectory);

  void update(const std::vector<laser_slam::Trajectory>& trajectories);

  void saveTimings() const {
    segmatch_.saveTimings();
  }
  
  void publish();
  
  void stopPublishing(unsigned int track_id) {
      publish_local_representation_[track_id] = false;
  }
  
 private:

  void loadTargetCloud();
  void publishTargetRepresentation() const;
  void publishSourceRepresentation() const;
  void publishTargetReconstruction() const;
  void publishSourceReconstruction() const;
  void publishSourceSemantics() const;
  void publishTargetSemantics() const;
  void publishMatches() const;
  void publishSegmentationPositions() const;
  void publishLoopClosures() const;
  void publishTargetSegmentsCentroids() const;
  void publishSourceSegmentsCentroids() const;
  void publishSourceBoundingBoxes() const;
  void publishTargetBoundingBoxes() const;
  bool exportRunServiceCall(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);
  bool reconstructSegmentsServiceCall(std_srvs::Empty::Request& req,
                                      std_srvs::Empty::Response& res);
  bool toggleCompressionServiceCall(std_srvs::Empty::Request& req,
                                    std_srvs::Empty::Response& res);
  bool togglePublishTargetServiceCall(std_srvs::Empty::Request& req,
                                      std_srvs::Empty::Response& res);
  bool exportTargetMap(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

  // Parameters.
  SegMatchWorkerParams params_;

  // Publishers.
  ros::Publisher source_representation_pub_;
  ros::Publisher source_reconstruction_pub_;
  ros::Publisher target_representation_pub_;
  ros::Publisher target_reconstruction_pub_;
  ros::Publisher source_semantics_pub_;
  ros::Publisher target_semantics_pub_;
  ros::Publisher matches_pub_;
  ros::Publisher predicted_matches_pub_;
  ros::Publisher loop_closures_pub_;
  ros::Publisher segmentation_positions_pub_;
  ros::Publisher target_segments_centroids_pub_;
  ros::Publisher source_segments_centroids_pub_;
  ros::Publisher reconstruction_pub_;
  ros::Publisher bounding_boxes_pub_;

  ros::ServiceServer export_run_service_;
  ros::ServiceServer reconstruct_segments_service_;
  ros::ServiceServer toggle_compression_service_;
  ros::ServiceServer toggle_publish_target_service_;
  ros::ServiceServer export_target_map_;

  // SegMatch object.
  segmatch::SegMatch segmatch_;

  bool target_cloud_loaded_ = false;

  typedef std::pair<laser_slam::Pose, unsigned int> PoseTrackIdPair;
  std::vector<PoseTrackIdPair> last_segmented_poses_;

  bool first_localization_occured_ = false;

  segmatch::SegmentedCloud segments_database_;
  segmatch::database::UniqueIdMatches matches_database_;

  std::unordered_map<unsigned int, segmatch::PointICloud> source_representations_;

  unsigned int num_tracks_;
  unsigned int pub_counter_ = 0;
  segmatch::PairwiseMatches matches_;
  
  std::vector<bool> publish_local_representation_;

  bool compress_when_publishing_ = false;
  bool publish_target_ = true;

  static constexpr unsigned int kPublisherQueueSize = 50u;
}; // SegMatchWorker

} // namespace segmatch_ros

#endif /* SEGMATCH_ROS_SEGMATCH_WORKER_HPP_ */
