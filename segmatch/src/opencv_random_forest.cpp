#include "segmatch/opencv_random_forest.hpp"

#include <laser_slam/benchmarker.hpp>
#include <laser_slam/common.hpp>
#include <ros/console.h>

using namespace Nabo;
using namespace Eigen;

namespace segmatch {

OpenCvRandomForest::OpenCvRandomForest(const ClassifierParams& params)
    : params_(params) {
  inverted_max_eigen_double_.resize(1, 7);
  inverted_max_eigen_float_.resize(1, 7);
  for (int i = 0; i < 7; ++i) {
    inverted_max_eigen_double_(0, i) = 1.0
        / params.max_eigen_features_values[i];
    inverted_max_eigen_float_(0, i) = float(
        1.0 / params.max_eigen_features_values[i]);
  }
}

OpenCvRandomForest::~OpenCvRandomForest() {
}

void OpenCvRandomForest::resetParams(const ClassifierParams& params) {
  LOG(INFO)<< "Reset classifier parameters.";
  LOG(INFO) << "n_nearest_neighbours: " << params_.n_nearest_neighbours;
  LOG(INFO) << "enable_two_stage_retrieval: " << params_.enable_two_stage_retrieval;
  LOG(INFO) << "knn_feature_dim: " << params_.knn_feature_dim;
  LOG(INFO) << "threshold_to_accept_match: " << params_.threshold_to_accept_match;
  LOG(INFO) << "classifier_filename: " << params_.classifier_filename;

  params_ = params;
}

void histogramIntersection(const Eigen::MatrixXd& h1, const Eigen::MatrixXd& h2,
                           Eigen::MatrixXd* intersection) {
  CHECK_EQ(h1.cols(), h2.cols());
  CHECK_EQ(h1.rows(), h2.rows());
  /*for (size_t i = 0u; i < h1.cols(); ++i) {
   CHECK_GE(h1(0, i), 0);
   CHECK_GE(h2(0, i), 0);

   intersection += std::min(h1(0, i), h2(0, i));
   }*/

  *intersection = (h1 + h2 - (h1 - h2).cwiseAbs()).rowwise().sum() / 2.0;
}

void OpenCvRandomForest::computeFeaturesDistance(const Eigen::MatrixXd& f1,
                                                 const Eigen::MatrixXd& f2,
                                                 Eigen::MatrixXd* f_out) const {
  CHECK_EQ(f1.cols(), f2.cols());
  CHECK_EQ(f1.rows(), f2.rows());
  const unsigned int n_sample = f1.rows();

  std::vector<Eigen::MatrixXd> fs;
  if (params_.descriptor_types.empty()) {
    *f_out = (f1 - f2).cwiseAbs();
  } else {
    unsigned int f_index = 0;
    unsigned int final_dim = 0;

    for (size_t i = 0u; i < params_.descriptor_types.size(); ++i) {
      if (params_.descriptor_types[i] == "EigenvalueBased") {
        const unsigned int f_dim = 7u;
        unsigned int f_dim_out = 7u;
        CHECK_GE(f1.cols(), f_index + f_dim);

        Eigen::MatrixXd v1 = f1.block(0, f_index, n_sample, f_dim);
        Eigen::MatrixXd v2 = f2.block(0, f_index, n_sample, f_dim);
        Eigen::MatrixXd f_diff = (v1 - v2).cwiseAbs();

        fs.push_back(f_diff);

        MatrixXd f1_abs = v1.cwiseAbs();
        MatrixXd f2_abs = v2.cwiseAbs();

        MatrixXd f_diff_norm_2 = f_diff.cwiseQuotient(f2_abs);
        MatrixXd f_diff_norm_1 = f_diff.cwiseQuotient(f1_abs);

        if (!params_.apply_hard_threshold_on_feature_distance) {
          // Augment the eigen feature vectors.
          fs.push_back(f_diff_norm_2);
          fs.push_back(f_diff_norm_1);
          fs.push_back(f1_abs);
          fs.push_back(f1_abs);
          f_dim_out = 35u;
        }

        f_index += f_dim;
        final_dim += f_dim_out;
      } else if (params_.descriptor_types[i] == "EnsembleShapeFunctions") {
        const unsigned int f_dim = 640u;
        const unsigned int f_dim_out = 10u;
        const unsigned int bin_size = 64u;
        CHECK_GE(f1.cols(), f_index + f_dim);

        Eigen::MatrixXd f(n_sample, f_dim_out);
        Eigen::MatrixXd h1 = f1.block(0, f_index, n_sample, f_dim);
        Eigen::MatrixXd h2 = f2.block(0, f_index, n_sample, f_dim);

        for (size_t i = 0; i < f_dim_out; ++i) {
          Eigen::MatrixXd intersection;
          histogramIntersection(h1.block(0, i * bin_size, n_sample, bin_size),
                                h2.block(0, i * bin_size, n_sample, bin_size),
                                &intersection);
          f.block(0, i, n_sample, 1) = intersection;
        }

        fs.push_back(f);
        f_index += f_dim;
        final_dim += f_dim_out;
      } else {
        CHECK(false) << "Distance not implemented.";
      }
    }

    // Reconstruct the feature vector.
    f_out->resize(n_sample, final_dim);
    f_index = 0;
    for (size_t i = 0u; i < fs.size(); ++i) {
      f_out->block(0, f_index, n_sample, fs[i].cols()) = fs[i];
      f_index += fs[i].cols();
    }
  }
}

PairwiseMatches OpenCvRandomForest::findCandidates(
    const SegmentedCloud& source_cloud,
    PairwiseMatches* matches_after_first_stage) {
  if (matches_after_first_stage != NULL) {
    matches_after_first_stage->clear();
  }
  PairwiseMatches candidates;
  PairwiseMatches candidates_after_first_stage;

  if (target_cloud_.empty()
      || target_cloud_.getNumberOfValidSegments()
          < kMinNumberSegmentInTargetCloud) {
    return candidates;
  }

  double time_in_compute_distance = 0;

  /*if (params_.n_nearest_neighbours > 0 && params_.enable_two_stage_retrieval) {
    if (params_.apply_hard_threshold_on_feature_distance) {
      LOG(INFO)<< "Two stage retrieval with hard threshold and " <<
      target_cloud_.getNumberOfValidSegments() << " segments in the target cloud and " <<
      source_cloud.getNumberOfValidSegments() << "  segments in the source cloud.";
    } else {
      LOG(INFO) << "Two stage retrieval with RF and " <<
      target_cloud_.getNumberOfValidSegments() << " segments in the target cloud and " <<
      source_cloud.getNumberOfValidSegments() << "  segments in the source cloud.";
    }
  } else if (params_.n_nearest_neighbours > 0) {
    LOG(INFO) << "Finding candidates with libnabo knn and " <<
    target_cloud_.getNumberOfValidSegments() << " segments in the target cloud and " <<
    source_cloud.getNumberOfValidSegments() << "  segments in the source cloud.";
  } else {
    LOG(INFO) << "Finding candidates with RF and " <<
    target_cloud_.getNumberOfValidSegments() << " segments in the target cloud and " <<
    source_cloud.getNumberOfValidSegments() << "  segments in the source cloud.";
  }*/

  if (params_.n_nearest_neighbours > 0) {
    for (std::unordered_map<Id, Segment>::const_iterator it_source = source_cloud.begin();
        it_source != source_cloud.end(); ++it_source) {

      if (params_.do_not_use_cars) {
        if (it_source->second.empty()) continue;  
        if (it_source->second.getLastView().semantic == 1u) continue;
      }

      Segment source_segment = it_source->second;
      Eigen::MatrixXd features_source = 
          source_segment.getLastView().features.rotationInvariantFeaturesOnly().asEigenMatrix();

      VectorXf q;
      if (params_.normalize_eigen_for_knn) {
        Eigen::MatrixXd features_source_normalized = features_source;
        normalizeEigenFeatures(&features_source_normalized);
        q = features_source_normalized.block(0, 0, 1, params_.knn_feature_dim)
            .transpose().cast<float>();
      } else {
        q = features_source.block(0, 0, 1, params_.knn_feature_dim).transpose()
            .cast<float>();
      }

      const unsigned int n_nearest_neighbours = std::min(
          params_.n_nearest_neighbours, int(target_segment_ids_.size()) - 1);
      VectorXi indices(n_nearest_neighbours);
      VectorXf dists2(n_nearest_neighbours);
      nns_->knn(q, indices, dists2, n_nearest_neighbours);

      for (size_t i = 0u; i < n_nearest_neighbours; ++i) {
        if (indices[i] == 0) {
          // TODO RD Sometimes all the indices are 0. Investigate this. 
          break;
        }
        if (source_segment.segment_id != target_segment_ids_[indices[i]]) {
          PairwiseMatch match(source_segment.segment_id,
                              target_segment_ids_[indices[i]],
                              source_segment.getLastView().centroid,
                              target_segment_centroids_[indices[i]], 1.0);
          match.features1_ = features_source;
          match.features2_ = target_segment_features_[indices[i]];

          candidates_after_first_stage.push_back(match);
        }
      }
    }

    if (matches_after_first_stage != NULL) {
      *matches_after_first_stage = candidates_after_first_stage;
    }

    if (params_.enable_two_stage_retrieval) {

      if (params_.apply_hard_threshold_on_feature_distance) {
        // Two stage knn and hard threshold.
        for (size_t i = 0u; i < candidates_after_first_stage.size(); ++i) {
          PairwiseMatch candidate = candidates_after_first_stage[i];
          Eigen::MatrixXd f1 = candidate.features1_;
          Eigen::MatrixXd f2 = candidate.features2_;
          /*if (params_.normalize_eigen_for_hard_threshold) {
            normalizeEigenFeatures(&f1);
            normalizeEigenFeatures(&f2);
          }*/

          if ((f1 - f2).squaredNorm() < params_.feature_distance_threshold) {
            candidates.push_back(candidate);
          }
        }
        /*LOG(INFO) << "candidates_after_first_stage.size() " <<
            candidates_after_first_stage.size() << " candidates.size() " <<
            candidates.size();*/
      } else {
        // Two stage knn and RF.
        CHECK(false) << "RF is not implemented anymore.";
      }
    } else {
      // knn only.
      candidates = candidates_after_first_stage;
    }
  } else {
    // RF only.
    CHECK(false) << "RF is not implemented anymore.";
  }

  //LOG(INFO)<< "Found " << candidates.size() << " candidates.";
  return candidates;
}

void OpenCvRandomForest::train(const Eigen::MatrixXd& features,
                               const Eigen::MatrixXd& labels) {
  CHECK(false) << "RF is not implemented anymore.";
}

void OpenCvRandomForest::test(const Eigen::MatrixXd& features,
                              const Eigen::MatrixXd& labels,
                              Eigen::MatrixXd* probabilities) const {
  CHECK(false) << "RF is not implemented anymore.";
}

void OpenCvRandomForest::save(const std::string& filename) const {
  CHECK(false) << "RF is not implemented anymore.";
}

void OpenCvRandomForest::load(const std::string& filename) {
  CHECK(false) << "RF is not implemented anymore.";
}

void OpenCvRandomForest::setTarget(const SegmentedCloud& target_cloud) {
  BENCHMARK_BLOCK("SM.Worker.UpdateTarget.SetClassifierTarget");
  if (target_cloud.empty()) {
    return;
  }

  target_cloud_ = target_cloud;

  target_segment_ids_.clear();
  target_segment_centroids_.clear();
  target_segment_features_.clear();

  // TODO RD Solve the need for cleaning empty segments and clean here.
  unsigned int n_non_empty_views = 0;
  for (std::unordered_map<Id, Segment>::const_iterator it = target_cloud.begin();
      it != target_cloud.end(); ++it) {
      if (!it->second.empty()) {
          ++n_non_empty_views;
      }
  }
  
  if (n_non_empty_views != target_cloud.getNumberOfValidSegments()) { 
      LOG(INFO) << "Some segments had empty views";
  }
  
  target_matrix_.resize(n_non_empty_views,
                        params_.knn_feature_dim);

  if (params_.do_not_use_cars) {
    // Find the number of segments which do not represent cars.
    unsigned int n_non_car = 0;
    for (std::unordered_map<Id, Segment>::const_iterator it = target_cloud.begin();
        it != target_cloud.end(); ++it) {
        if (!it->second.empty()) {
            if(it->second.getLastView().semantic != 1u) {
                ++n_non_car;
            } 
        }
    }
    target_matrix_.resize(n_non_car,
                          params_.knn_feature_dim);
  }

  unsigned int i = 0u;
  for (std::unordered_map<Id, Segment>::const_iterator it = target_cloud.begin();
      it != target_cloud.end(); ++it) {
    Segment target_segment = it->second;
    if (target_segment.empty()) continue;
    if (params_.do_not_use_cars && target_segment.getLastView().semantic == 1u) continue;

    target_matrix_.block(i, 0, 1, params_.knn_feature_dim) =
        target_segment.getLastView().features.rotationInvariantFeaturesOnly().asEigenMatrix().block(
            0, 0, 1, params_.knn_feature_dim).cast<float>();
    target_segment_ids_.push_back(target_segment.segment_id);
    target_segment_centroids_.push_back(target_segment.getLastView().centroid);
    target_segment_features_.push_back(
        target_segment.getLastView().features.rotationInvariantFeaturesOnly().asEigenMatrix());
    ++i;
  }

  if (params_.normalize_eigen_for_knn) {
    normalizeEigenFeatures(&target_matrix_);
  }

  target_matrix_.transposeInPlace();
  nns_ = NNSearchF::createKDTreeLinearHeap(target_matrix_);
}

void OpenCvRandomForest::normalizeEigenFeatures(Eigen::MatrixXd* f) {
  for (size_t i = 0u; i < f->rows(); ++i) {
    f->block(i, 0, 1, 7) = f->block(i, 0, 1, 7).cwiseProduct(
        inverted_max_eigen_double_);
  }
}

void OpenCvRandomForest::normalizeEigenFeatures(Eigen::MatrixXf* f) {
  for (size_t i = 0u; i < f->rows(); ++i) {
    f->block(i, 0, 1, 7) = f->block(i, 0, 1, 7).cwiseProduct(
        inverted_max_eigen_float_);
  }
}

}  // namespace segmatch
