#include "segmatch/opencv_random_forest.hpp"

#include <laser_slam/common.hpp>

using namespace Nabo;
using namespace Eigen;
using namespace cv;
using namespace segmatch;

void convertEigenToOpenCvMat(const Eigen::MatrixXd& eigen_mat,
                             Mat* open_cv_mat) {
  CHECK_NOTNULL(open_cv_mat);
  for (size_t i = 0u; i < eigen_mat.rows(); ++i) {
    for (size_t j = 0u; j < eigen_mat.cols(); ++j) {
      open_cv_mat->at<float>(i, j) = eigen_mat(i, j);
    }
  }
}

namespace segmatch {

OpenCvRandomForest::OpenCvRandomForest(const ClassifierParams& params)
    : params_(params) {
  rtrees_.load(params.classifier_filename.c_str());

  inverted_max_eigen_double_.resize(1, 7);
  inverted_max_eigen_float_.resize(1, 7);
  for (int i = 0; i < 7; ++i) {
    inverted_max_eigen_double_(0, i) = 1.0
        / params.max_eigen_features_values[i];
    inverted_max_eigen_float_(0, i) = float(
        1.0 / params.max_eigen_features_values[i]);
  }
  std::cout << "inverted_max_eigen_double_ " << std::endl
            << inverted_max_eigen_double_ << std::endl;
  std::cout << "inverted_max_eigen_float_ " << std::endl
            << inverted_max_eigen_float_ << std::endl;
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
  rtrees_.load(params.classifier_filename.c_str());
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
  laser_slam::Clock clock;
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

  if (params_.n_nearest_neighbours > 0 && params_.enable_two_stage_retrieval) {
    if (params_.apply_hard_threshold_on_feature_distance) {
      LOG(INFO)<< "Two stage retrieval with hard threshold and " <<
      target_cloud_.getNumberOfValidSegments() << " segments in the target cloud.";
    } else {
      LOG(INFO) << "Two stage retrieval with RF and " <<
      target_cloud_.getNumberOfValidSegments() << " segments in the target cloud.";
    }
  } else if (params_.n_nearest_neighbours > 0) {
    LOG(INFO) << "Finding candidates with libnabo knn and " <<
    target_cloud_.getNumberOfValidSegments() << " segments in the target cloud.";
  } else {
    LOG(INFO) << "Finding candidates with RF and " <<
    target_cloud_.getNumberOfValidSegments() << " segments in the target cloud.";
  }

  if (params_.n_nearest_neighbours > 0) {

    for (std::unordered_map<Id, Segment>::const_iterator it_source = source_cloud.begin();
        it_source != source_cloud.end(); ++it_source) {
      Segment source_segment = it_source->second;
      Eigen::MatrixXd features_source = source_segment.features.asEigenMatrix();

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
        PairwiseMatch match(source_segment.segment_id,
                            target_segment_ids_[indices[i]],
                            source_segment.centroid,
                            target_segment_centroids_[indices[i]], 1.0);
        match.features1_ = features_source;
        match.features2_ = target_segment_features_[indices[i]];

        candidates_after_first_stage.push_back(match);
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
          if (params_.normalize_eigen_for_hard_threshold) {
            normalizeEigenFeatures(&f1);
            normalizeEigenFeatures(&f2);
          }

          if ((f1 - f2).squaredNorm() < params_.feature_distance_threshold) {
            candidates.push_back(candidate);
          }
        }

      } else {
        // Two stage knn and RF.

        const unsigned int feature_dimension = candidates_after_first_stage[0]
            .features1_.cols();
        Eigen::MatrixXd F1(candidates_after_first_stage.size(),
                           feature_dimension);
        Eigen::MatrixXd F2(candidates_after_first_stage.size(),
                           feature_dimension);

        for (size_t i = 0u; i < candidates_after_first_stage.size(); ++i) {
          PairwiseMatch candidate = candidates_after_first_stage[i];
          F1.block(i, 0, 1, feature_dimension) = candidate.features1_;
          F2.block(i, 0, 1, feature_dimension) = candidate.features2_;
        }

        Eigen::MatrixXd dF;
        computeFeaturesDistance(F1, F2, &dF);
        const unsigned int feature_dim_after_dist = dF.cols();

        for (size_t i = 0u; i < candidates_after_first_stage.size(); ++i) {
          PairwiseMatch candidate = candidates_after_first_stage[i];
          Mat features(1, feature_dim_after_dist, CV_32FC1);
          convertEigenToOpenCvMat(dF.block(i, 0, 1, feature_dim_after_dist),
                                  &features);
          const float confidence = rtrees_.predict_prob(features);
          if (confidence > params_.threshold_to_accept_match) {
            candidates.push_back(
                PairwiseMatch(candidate.ids_.first, candidate.ids_.second,
                              candidate.centroids_.first,
                              candidate.centroids_.second, confidence));
          }
        }

        //        for (size_t i = 0u; i < candidates_after_first_stage.size(); ++i) {
        //          PairwiseMatch candidate = candidates_after_first_stage[i];
        //          Eigen::MatrixXd diff_features;
        //          computeFeaturesDistance(candidate.features1_, candidate.features2_, &diff_features);
        //
        //          Mat features(1, diff_features.cols(), CV_32FC1);
        //          convertEigenToOpenCvMat(diff_features, &features);
        //
        //          const float confidence = rtrees_.predict_prob(features);
        //          if (confidence > params_.threshold_to_accept_match) {
        //            candidates.push_back(PairwiseMatch(candidate.ids_.first, candidate.ids_.second,
        //                                               candidate.centroids_.first, candidate.centroids_.second,
        //                                               confidence));
        //          }
        //        }
      }
    } else {
      // knn only.
      candidates = candidates_after_first_stage;
    }
  } else {
    // RF only.
    for (std::unordered_map<Id, Segment>::const_iterator it_source = source_cloud.begin();
        it_source != source_cloud.end(); ++it_source) {
      Segment source_segment = it_source->second;
      Eigen::MatrixXd features_source = source_segment.features.asEigenMatrix();

      for (std::unordered_map<Id, Segment>::const_iterator it_target = target_cloud_.begin();
          it_target != source_cloud.end(); ++it_target) {
        Segment target_segment = it_target->second;
        Eigen::MatrixXd features_target =
            target_segment.features.asEigenMatrix();

        Eigen::MatrixXd diff_features;
        computeFeaturesDistance(features_source, features_target,
                                &diff_features);

        Mat features(1, diff_features.cols(), CV_32FC1);
        convertEigenToOpenCvMat(diff_features, &features);

        const float confidence = rtrees_.predict_prob(features);
        if (confidence > params_.threshold_to_accept_match) {
          candidates.push_back(
              PairwiseMatch(source_segment.segment_id,
                            target_segment.segment_id, source_segment.centroid,
                            target_segment.centroid, confidence));
        }
      }
    }
  }

  clock.takeTime();
  LOG(INFO)<< "Found " << candidates.size() << " candidates in "
  << clock.getRealTime() << "ms with " << time_in_compute_distance <<
  " ms in computing distance." << std::endl;
  return candidates;
}

void OpenCvRandomForest::train(const Eigen::MatrixXd& features,
                               const Eigen::MatrixXd& labels) {
  const unsigned int n_training_samples = features.rows();
  const unsigned int descriptors_dimension = features.cols();
  LOG(INFO)<< "Training RF with " << n_training_samples << " of dimension "
  << descriptors_dimension << ".";

  Mat opencv_features(n_training_samples, descriptors_dimension, CV_32FC1);
  Mat opencv_labels(n_training_samples, 1, CV_32FC1);
  //TODO Functionalize.
  for (unsigned int i = 0u; i < n_training_samples; ++i) {
    for (unsigned int j = 0u; j < descriptors_dimension; ++j) {
      opencv_features.at<float>(i, j) = features(i, j);
    }
    opencv_labels.at<float>(i, 0) = labels(i, 0);
  }

  float priors[] = { params_.rf_priors[0], params_.rf_priors[1] };

  // Random forest parameters.
  CvRTParams rtrees_params = CvRTParams(
      params_.rf_max_depth, params_.rf_min_sample_ratio * n_training_samples,
      params_.rf_regression_accuracy, params_.rf_use_surrogates,
      params_.rf_max_categories, priors, params_.rf_calc_var_importance,
      params_.rf_n_active_vars, params_.rf_max_num_of_trees,
      params_.rf_accuracy,
      CV_TERMCRIT_EPS);

  rtrees_.train(opencv_features, CV_ROW_SAMPLE, opencv_labels, cv::Mat(),
                cv::Mat(), cv::Mat(), cv::Mat(), rtrees_params);
  ROS_INFO_STREAM("Tree count: " << rtrees_.get_tree_count() << ".");

  if (params_.rf_calc_var_importance) {
    Mat variable_importance = rtrees_.getVarImportance();
    Size variable_importance_size = variable_importance.size();
    CHECK_EQ(variable_importance_size.height, 1.0)<< "Height of variable importance is not 1.";
    CHECK_EQ(variable_importance_size.width, descriptors_dimension)<<
    "Width of variable importance is the features dimension.";

    // TODO(renaud): Remove this cout (just for debugging).
    std::cout << "Variable importance: ";
    for (unsigned int i = 0; i < descriptors_dimension; ++i) {
      std::cout << variable_importance.at<float>(i) << " ";
    }
    std::cout << std::endl;
  }

  // TODO if desired, re-implement negative mining.
}

void OpenCvRandomForest::test(const Eigen::MatrixXd& features,
                              const Eigen::MatrixXd& labels,
                              Eigen::MatrixXd* probabilities) const {
  laser_slam::Clock clock;
  const unsigned int n_samples = features.rows();
  const unsigned int descriptors_dimension = features.cols();
  LOG(INFO)<< "Testing the random forest with " << n_samples
  << " samples of dimension " << descriptors_dimension << ".";

  if (probabilities != NULL) {
    probabilities->resize(n_samples, 1);
  }

  if (n_samples > 0u) {
    unsigned int tp = 0u, fp = 0u, tn = 0u, fn = 0u;
    for (unsigned int i = 0u; i < n_samples; ++i) {
      Mat opencv_sample(1, descriptors_dimension, CV_32FC1);
      for (unsigned int j = 0u; j < descriptors_dimension; ++j) {
        opencv_sample.at<float>(j) = features(i, j);
      }
      double probability = rtrees_.predict_prob(opencv_sample);
      if (probability >= params_.threshold_to_accept_match) {
        if (labels(i, 0) == 1.0) {
          ++tp;
        } else {
          ++fp;
        }
      } else {
        if (labels(i, 0) == 0.0) {
          ++tn;
        } else {
          ++fn;
        }
      }
      if (probabilities != NULL) {
        (*probabilities)(i, 0) = probability;
      }
    }
    displayPerformances(tp, tn, fp, fn);
  }
  clock.takeTime();
  LOG(INFO)<< "Took " << clock.getRealTime() << "ms to test.";
}

void OpenCvRandomForest::save(const std::string& filename) const {
  LOG(INFO)<< "Saving the classifier to: " << filename << ".";
  rtrees_.save(filename.c_str());
}

void OpenCvRandomForest::load(const std::string& filename) {
  LOG(INFO)<< "Loading a classifier from: " << filename << ".";
  rtrees_.load(filename.c_str());
}

void OpenCvRandomForest::setTarget(const SegmentedCloud& target_cloud) {
  if (target_cloud.empty()) {
    return;
  }

  target_cloud_ = target_cloud;

  target_segment_ids_.clear();
  target_segment_centroids_.clear();
  target_segment_features_.clear();

  target_matrix_.resize(target_cloud.getNumberOfValidSegments(),
                        params_.knn_feature_dim);

  unsigned int i = 0u;
  for (std::unordered_map<Id, Segment>::const_iterator it = target_cloud.begin();
      it != target_cloud.end(); ++it) {
    Segment target_segment = it->second;
    target_matrix_.block(i, 0, 1, params_.knn_feature_dim) = target_segment
        .features.asEigenMatrix().block(0, 0, 1, params_.knn_feature_dim)
        .cast<float>();
    target_segment_ids_.push_back(target_segment.segment_id);
    target_segment_centroids_.push_back(target_segment.centroid);
    target_segment_features_.push_back(target_segment.features.asEigenMatrix());
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
