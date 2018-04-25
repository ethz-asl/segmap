#include "segmatch/features.hpp"

namespace segmatch {

bool Feature::findValueByName(const std::string& name, FeatureValue* value) const {
  for (size_t i = 0u; i < feature_values_.size(); ++i) {
    if (feature_values_.at(i).name == name) {
      if(value != NULL) {
        *value = feature_values_.at(i);
      }
      return true;
    }
  }
  return false;
}

size_t Features::sizeWhenFlattened() const {
  size_t result = 0u;
  for (size_t i = 0u; i < size(); ++i) {
    result += at(i).size();
  }
  return result;
}

std::vector<FeatureValueType> Features::asVectorOfValues() const {
  std::vector<FeatureValueType> result;
  for (size_t i = 0u; i < size(); ++i) {
    for (size_t j = 0u; j < at(i).size(); j++) {
      result.push_back(at(i).at(j).value);
    }
  }
  return result;
}

Eigen::MatrixXd Features::asEigenMatrix() const {
  std::vector<FeatureValueType> as_vector = asVectorOfValues();
  Eigen::MatrixXd matrix(1, as_vector.size());
  for (size_t i = 0u; i < as_vector.size(); ++i) {
    matrix(0,i) = as_vector.at(i);
  }
  return matrix;
}

Features Features::rotationInvariantFeaturesOnly() const {
  Features result;
  for (size_t i = 0u; i < size(); ++i) {
    Feature feature;
    for (size_t j = 0u; j < at(i).size(); j++) {
      if (at(i).at(j).name != "scale_x" &&
          at(i).at(j).name != "scale_y" &&
          at(i).at(j).name != "scale_z" &&
          at(i).at(j).name != "scale_sml" &&
          at(i).at(j).name != "scale_med" &&
          at(i).at(j).name != "scale_lrg" &&
          at(i).at(j).name != "alignment" &&
          at(i).at(j).name != "origin_dx" &&
          at(i).at(j).name != "origin_dy") {
        feature.push_back(at(i).at(j));
      }
    }
    if (!feature.empty()) { result.push_back(feature); }
  }
  return result;
}

std::vector<std::string> Features::asVectorOfNames() const {
  std::vector<std::string> result;
  for (size_t i = 0u; i < size(); ++i) {
    for (size_t j = 0u; j < at(i).size(); j++) {
      result.push_back(at(i).at(j).name);
    }
  }
  return result;
}

void Features::clearByName(const std::string& name) {
  auto removal_predicate = [&](const Feature& feature) {
    return feature.getName() == name;
  };
  features_.erase(std::remove_if(features_.begin(), features_.end(), removal_predicate),
                  features_.end());
}

void Features::replaceByName(const Feature& new_feature) {
  bool found = false;
  for (auto& feature : features_) {
    if (feature.getName() == new_feature.getName()) {
      found = true;
      feature = new_feature;
      break;
    }
  }
  if (!found) features_.push_back(new_feature);
}



} // namespace segmatch
