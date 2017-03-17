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
          at(i).at(j).name != "alignment") {
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

} // namespace segmatch
