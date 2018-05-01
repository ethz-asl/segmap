#ifndef SEGMATCH_PARAMETERS_HPP_
#define SEGMATCH_PARAMETERS_HPP_

#include <string>
#include <vector>

namespace segmatch {

// TODO: Check that parameter values are reasonable when setting them.

struct KeypointSelectionParams {
  std::string keypoint_selection;
  double uniform_sample_size;
  int minimum_point_number_per_voxel;
  double harris_threshold;
  double minimum_keypoint_distance;
}; // struct KeypointSelectionParams

struct DescriptorsParameters {
  std::vector<std::string> descriptor_types;

  // FastPointFeatureHistograms parameters.
  double fast_point_feature_histograms_search_radius = 0.8;
  double fast_point_feature_histograms_normals_search_radius = 0.5;

  // PointFeatureHistograms parameters.
  double point_feature_histograms_search_radius = 0.8;
  double point_feature_histograms_normals_search_radius = 0.5;

  // CNN parameters.
  std::string cnn_model_path = "MUST_BE_SET";
  std::string semantics_nn_path = "MUST_BE_SET";
}; // struct DescriptorsParameters

struct SegmenterParameters {
  // Region growing segmenter parameters.
  std::string segmenter_type = "IncrementalEuclideanDistance";
  int min_cluster_size;
  int max_cluster_size;
  float radius_for_growing;

  // Parameters specific for the SmoothnessConstraint growing policy.
  float sc_smoothness_threshold_deg;
  float sc_curvature_threshold;
}; // struct SegmenterParameters

struct ClassifierParams {
  std::string classifier_filename;
  double threshold_to_accept_match;

  // OpenCv random forest parameters.
  int rf_max_depth;
  double rf_min_sample_ratio;
  double rf_regression_accuracy;
  bool rf_use_surrogates;
  int rf_max_categories;
  std::vector<double> rf_priors;
  bool rf_calc_var_importance;
  int rf_n_active_vars;
  int rf_max_num_of_trees;
  double rf_accuracy;

  // A convenience copy from DescriptorsParameters.
  std::vector<std::string> descriptor_types;

  int n_nearest_neighbours;
  bool enable_two_stage_retrieval;
  int knn_feature_dim;
  bool apply_hard_threshold_on_feature_distance;
  double feature_distance_threshold;

  bool normalize_eigen_for_knn;
  bool normalize_eigen_for_hard_threshold;
  std::vector<double> max_eigen_features_values;

  bool do_not_use_cars;

}; // struct ClassifierParams

struct CorrespondeceParams {
  std::string matching_method;
  double corr_sqr_dist_thresh = 500.0;
  int n_neighbours;
}; // struct CorrespondeceParams

struct GeometricConsistencyParams {
  // Type of recognizer.
  std::string recognizer_type;
  // Higher resolutions lead to higher tolerances.
  double resolution = 0.2;
  // Minimum number of matches necessary to consider cluster.
  int min_cluster_size = 10;
  // Maximum consistency distance between two matches in order for them to be cached as candidates.
  // Used in the incremental recognizer only.
  float max_consistency_distance_for_caching = 10.0f;
}; // struct GeometricConsistencyParams

struct GroundTruthParameters {
  double overlap_radius = 0.4;
  double significance_percentage = 50.0;
  int number_nearest_segments;
  double maximum_centroid_distance_m;
}; // struct GroundTruthParameters

struct Parameters {
  DescriptorsParameters descriptors_parameters;
  SegmenterParameters segmenter_parameters;
  GroundTruthParameters ground_truth_parameters;
};

} // namespace segmatch

#endif // SEGMATCH_PARAMETERS_HPP_
