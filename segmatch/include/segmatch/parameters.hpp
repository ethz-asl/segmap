#ifndef SEGMATCH_PARAMETERS_HPP_
#define SEGMATCH_PARAMETERS_HPP_

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

  // Autoencoder parameters.
  std::string autoencoder_python_env = "python"; // Example: "home/your_profile/anaconda2/envs/ml/bin/python"
  std::string autoencoder_script_path = "MUST_BE_SET";
  std::string autoencoder_model_path = "MUST_BE_SET";
  std::string autoencoder_temp_folder_path = "/tmp/";
  int autoencoder_latent_space_dimension = 10;
}; // struct DescriptorsParameters

struct SegmenterParameters {
  std::string segmenter_type = "DonSegmenter";

  // DonSegmenter parameters.
  double don_segmenter_small_scale = 0.5;
  double don_segmenter_large_scale = 2;
  double don_segmenter_don_threshold = 0.2;
  double don_segmenter_distance_tolerance = 1.0;

  // Region growing parameters.
  int rg_min_cluster_size;
  int rg_max_cluster_size;
  int rg_knn_for_normals;
  double rg_radius_for_normals;
  int rg_knn_for_growing;
  double rg_smoothness_threshold_deg;
  double rg_curvature_threshold;

  // Euclidean segmenter parameters.
  double ec_tolerance;
  int ec_max_cluster_size;
  int ec_min_cluster_size;
}; // struct SegmenterParameters

struct ClassifierParams {
  std::string classifier_filename;
  double threshold_to_accept_match;

  // OpenCv random forest parameters.
  int rf_max_depth;
  double rf_min_sample_ratio;
  float rf_regression_accuracy;
  bool rf_use_surrogates;
  int rf_max_categories;
  std::vector<float> rf_priors;
  bool rf_calc_var_importance;
  int rf_n_active_vars;
  int rf_max_num_of_trees;
  float rf_accuracy;

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

}; // struct ClassifierParams

struct CorrespondeceParams {
  std::string matching_method;
  double corr_sqr_dist_thresh = 500.0;
  int n_neighbours;
}; // struct CorrespondeceParams

struct GeometricConsistencyParams {
  // Higher resolutions lead to higher tolerances.
  double resolution = 0.2;
  // Minimum number of matches necessary to consider cluster.
  int min_cluster_size = 10;
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
