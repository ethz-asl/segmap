#ifndef SEGMATCH_ROS_SEGMATCH_WORKER_HPP_
#define SEGMATCH_ROS_SEGMATCH_WORKER_HPP_

#include <utility>

#include <laser_slam/common.hpp>
#include <segmatch/database.hpp>
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

  // Process the source cloud and return true if a loop closure was found.
  bool processSourceCloud(const segmatch::PointICloud& source_cloud,
                          const laser_slam::Pose& latest_pose,
                          unsigned int track_id = 0u,
                          laser_slam::RelativePose* loop_closure = NULL);

  void update(const laser_slam::Trajectory& trajectory);

  void update(const std::vector<laser_slam::Trajectory>& trajectories);

 private:

  void loadTargetCloud();

  void publish() const;
  void publishTargetRepresentation() const;
  void publishSourceRepresentation() const;
  void publishMatches() const;
  void publishSegmentationPositions() const;
  void publishLoopClosures() const;
  void publishTargetSegmentsCentroids() const;
  void publishSourceSegmentsCentroids() const;
  void publishLastTransformation() const;

  bool exportRunServiceCall(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);
  bool reconstructSegmentsServiceCall(std_srvs::Empty::Request& req,
                                      std_srvs::Empty::Response& res);

  // Parameters.
  SegMatchWorkerParams params_;

  // Publishers.
  ros::Publisher source_representation_pub_;
  ros::Publisher target_representation_pub_;
  ros::Publisher matches_pub_;
  ros::Publisher predicted_matches_pub_;
  ros::Publisher loop_closures_pub_;
  ros::Publisher segmentation_positions_pub_;
  ros::Publisher target_segments_centroids_pub_;
  ros::Publisher source_segments_centroids_pub_;
  ros::Publisher last_transformation_pub_;
  ros::Publisher reconstruction_pub_;

  ros::ServiceServer export_run_service_;
  ros::ServiceServer reconstruct_segments_service_;

  // SegMatch object.
  segmatch::SegMatch segmatch_;

  bool target_cloud_loaded_ = false;

  typedef std::pair<laser_slam::Pose, unsigned int> PoseTrackIdPair;
  std::vector<PoseTrackIdPair> last_segmented_poses_;

  bool first_localization_occured = false;

  segmatch::SegmentedCloud segments_database_;
  segmatch::database::UniqueIdMatches matches_database_;

  // Publishing parameters.
  static constexpr float kLineScaleSegmentMatches = 0.3;
  static constexpr float kLineScaleLoopClosures = 3.0;

  static constexpr unsigned int kPublisherQueueSize = 50u;
}; // SegMatchWorker

} // namespace segmatch_ros

#endif /* SEGMATCH_ROS_SEGMATCH_WORKER_HPP_ */
