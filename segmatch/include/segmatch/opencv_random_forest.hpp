#ifndef SEGMATCH_OPENCV_RANDOM_FOREST_HPP_
#define SEGMATCH_OPENCV_RANDOM_FOREST_HPP_

#include <nabo/nabo.h>

#include "segmatch/common.hpp"
#include "segmatch/parameters.hpp"
#include "segmatch/segmented_cloud.hpp"

namespace segmatch {

//TODO Renaud matching needs major refactoring.
class OpenCvRandomForest {
 public:
  explicit OpenCvRandomForest(const ClassifierParams& params);
  ~OpenCvRandomForest();

  /// \brief Find candidates the source and target clouds.
  PairwiseMatches findCandidates(const SegmentedCloud& source_cloud,
                                 PairwiseMatches* matches_after_first_stage = NULL);

  /// \brief Compute the features distance.
  void computeFeaturesDistance(const Eigen::MatrixXd& f1, const Eigen::MatrixXd& f2,
                               Eigen::MatrixXd* f_out) const;

  void train(const Eigen::MatrixXd& features, const Eigen::MatrixXd& labels);

  void test(const Eigen::MatrixXd& features, const Eigen::MatrixXd& labels,
            Eigen::MatrixXd* probabilities = NULL) const;

  void save(const std::string& filename) const;

  void load(const std::string& filename);

  void setTarget(const SegmentedCloud& target_cloud);

  void resetParams(const ClassifierParams& params);

  void normalizeEigenFeatures(Eigen::MatrixXd* f);

  void normalizeEigenFeatures(Eigen::MatrixXf* f);

 private:
  std::vector<Id> target_segment_ids_;
  std::vector<PclPoint> target_segment_centroids_;
  std::vector<Eigen::MatrixXd> target_segment_features_;

  SegmentedCloud target_cloud_;

  Eigen::MatrixXf target_matrix_;
  Nabo::NNSearchF* nns_ = NULL;

  Eigen::MatrixXd inverted_max_eigen_double_;
  Eigen::MatrixXf inverted_max_eigen_float_;

  ClassifierParams params_;

  static constexpr unsigned int kMinNumberSegmentInTargetCloud = 50u;
}; // class OpenCvRandomForest

} // namespace segmatch

#endif // SEGMATCH_OPENCV_RANDOM_FOREST_HPP_
