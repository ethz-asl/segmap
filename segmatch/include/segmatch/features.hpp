#ifndef SEGMATCH_FEATURES_HPP_
#define SEGMATCH_FEATURES_HPP_

#include <string>
#include <vector>

#include <Eigen/Core>
#include <glog/logging.h>

namespace segmatch {

typedef double FeatureValueType;

struct FeatureValue {
  FeatureValue() {}
  FeatureValue(std::string feature_name, FeatureValueType feature_value) :
    name(feature_name), value(feature_value) {}
  std::string name = "";
  FeatureValueType value = 0.0;
};

/// \brief A feature can be composed of any number of values, each with its own name.
class Feature {
 public:
  Feature() {}
  ~Feature() {}
  explicit Feature(const std::string& name) : name_(name) {}

  size_t size() const { return feature_values_.size(); }
  bool empty() { return feature_values_.empty(); }
  const FeatureValue& at(const size_t& index) const { return feature_values_.at(index); }
  void clear() { feature_values_.clear(); }
  void push_back(const FeatureValue& value) {
    LOG_IF(WARNING, findValueByName(value.name, NULL)) <<
        "Adding several FeatureValues of same name to Feature is not recommended.";
    feature_values_.push_back(value);
  }

  bool findValueByName(const std::string& name, FeatureValue* value) const;

  std::string getName() const {return name_;}

 private:
  std::vector<FeatureValue> feature_values_;
  std::string name_;
}; // class Feature

/// \brief A collection of features.
class Features {
 public:
  Features() {}
  Features& operator+= (const Features& rhs) {
    this->features_.insert(this->features_.end(), rhs.features_.begin(), rhs.features_.end());
    return *this;
  }
  void push_back(const Feature& feature) { features_.push_back(feature); }
  size_t size() const { return features_.size(); }
  const Feature& at(const size_t& index) const { return features_.at(index); }
  void clear() { features_.clear(); }
  void clearByName(const std::string& name);
  bool empty() { return features_.empty(); }
  void replaceByName(const Feature& new_feature);

  size_t sizeWhenFlattened() const;
  std::vector<FeatureValueType> asVectorOfValues() const;
  Eigen::MatrixXd asEigenMatrix() const;
  Features rotationInvariantFeaturesOnly() const;
  std::vector<std::string> asVectorOfNames() const;

 private:
  std::vector<Feature> features_;
}; // class Features

} // namespace segmatch

#endif // SEGMATCH_FEATURES_HPP_
