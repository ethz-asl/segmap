// This code was originally written by Martin Pecka ( martin.pecka@cvut.cz ) and adapted
// for our application.

#include <utility>

#include <tf_graph_executor/tf_graph_executor.hpp>

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

using namespace std;
using namespace tensorflow;

namespace tf_graph_executor {

TensorflowGraphExecutor::TensorflowGraphExecutor(const std::string& pathToGraph) {
  LOG(INFO) << "Entering TensorflowGraphExecutor with path " << pathToGraph;
  this->tensorflowSession = NewSession(SessionOptions());
  if (this->tensorflowSession == nullptr) {
    throw runtime_error("Could not create Tensorflow session.");
  }

  // Read in the protobuf graph we exported
  Status status;
  this->graph_def = new MetaGraphDef;
  LOG(INFO) << "calling custom function";
  status = ReadBinaryProto(Env::Default(), pathToGraph, this->graph_def);
  LOG(INFO) << "Custom function called";
  if (!status.ok()) {
    throw runtime_error("Error reading graph definition from " + pathToGraph + ".");
  }

  // Add the graph to the session
  status = this->tensorflowSession->Create(this->graph_def->graph_def());
  if (!status.ok()) {
    throw runtime_error("Error creating graph.");
  }
}

void TensorflowGraphExecutor::loadCheckpoint(const string &checkpointPath) {
  LOG(INFO) << "Loading checkpoint " << checkpointPath;
  Status status;
  // Read weights from the saved checkpoint
  Tensor checkpointPathTensor(DT_STRING, TensorShape());
  checkpointPathTensor.scalar<std::string>()() = checkpointPath;
  status = this->tensorflowSession->Run(
      {{ this->graph_def->saver_def().filename_tensor_name(), checkpointPathTensor },},
      {},
      {this->graph_def->saver_def().restore_op_name()},
      nullptr);
  if (!status.ok()) {
    throw runtime_error("Error loading checkpoint from " + checkpointPath + ".");
  }
}

std::vector<float> TensorflowGraphExecutor::executeGraph(
    const std::vector<float>& inputs, const std::string& input_tensor_name,
    const std::string& output_tensor_name) const {
  auto session = this->tensorflowSession;
  Status status;

  // define the input placeholder
  auto inputShape = TensorShape();
  inputShape.AddDim(1);
  inputShape.AddDim((int64) inputs.size());
  Tensor inputTensor(DT_FLOAT, inputShape);

  // put values in the input tensor
  auto inputTensorValues = inputTensor.tensor<float, 2>();
  for (int i=0; i < inputs.size(); i++) {
    inputTensorValues(0, i) = inputs[i];
  }

  Tensor outputTensor;
  status = executeGraph(inputTensor, outputTensor, input_tensor_name, output_tensor_name);
  if (!status.ok()) {
    throw runtime_error("Error running inference in graph.");
  }

  auto outputTensorValues = outputTensor.tensor<float, 2>();

  // Pass the results to the output output vector
  std::vector<float> output;
  for (int i = 0; i < outputTensorValues.dimension(1); i++) {
    output.push_back(outputTensorValues(0, i));
  }

  return output;
}

std::vector<std::vector<float> > TensorflowGraphExecutor::batchExecuteGraph(
    const std::vector<std::vector<float> >& inputs, const std::string& input_tensor_name,
    const std::string& output_tensor_name) const {
  CHECK(!inputs.empty());
  Status status;

  // define the input placeholder
  auto inputShape = TensorShape();
  inputShape.AddDim((int64) inputs.size());
  inputShape.AddDim((int64) inputs[0].size());
  Tensor inputTensor(DT_FLOAT, inputShape);

  // put values in the input tensor
  auto inputTensorValues = inputTensor.tensor<float, 2>();
  for (int i=0; i < inputs.size(); i++) {
    for (size_t j=0; j < inputs[0].size(); j++) {
      inputTensorValues(i, j) = inputs[i][j];
    }
  }

  Tensor outputTensor;
  status = executeGraph(inputTensor, outputTensor, input_tensor_name, output_tensor_name);
  if (!status.ok()) {
    throw runtime_error("Error running inference in graph.");
  }

  auto outputTensorValues = outputTensor.tensor<float, 2>();

  // Pass the results to the output output vector
  std::vector<std::vector<float> > batch_output;
  for (int i = 0; i < outputTensorValues.dimension(0); i++) {
    std::vector<float> output;
    for (int j = 0; j < outputTensorValues.dimension(1); j++) {
      output.push_back(outputTensorValues(i, j));
    }
    batch_output.push_back(output);
  }

  return batch_output;
}

std::vector<std::vector<float> > TensorflowGraphExecutor::batchExecuteGraph(
    const std::vector<Array3D>& inputs, const std::string& input_tensor_name,
    const std::string& output_tensor_name) const {
  CHECK(!inputs.empty());
  Status status;

  // define the input placeholder
  auto inputShape = TensorShape();
  inputShape.AddDim((int64) inputs.size());
  inputShape.AddDim((int64) inputs[0].container.size());
  inputShape.AddDim((int64) inputs[0].container[0].size());
  inputShape.AddDim((int64) inputs[0].container[0][0].size());
  inputShape.AddDim((int64) 1u);
  Tensor inputTensor(DT_FLOAT, inputShape);

  // put values in the input tensor
  auto inputTensorValues = inputTensor.tensor<float, 5>();
  for (size_t i=0; i < inputs.size(); i++) {
    for (size_t j=0; j < inputs[0].container.size(); j++) {
      for (size_t k=0; k < inputs[0].container[0].size(); k++) {
        for (size_t l=0; l < inputs[0].container[0][0].size(); l++) {
          inputTensorValues(i, j, k, l, 0) = inputs[i].container[j][k][l];
        }
      }
    }
  }

  Tensor outputTensor;
  status = executeGraph(inputTensor, outputTensor, input_tensor_name, output_tensor_name);
  if (!status.ok()) {
    LOG(INFO) << status.error_message();
    throw runtime_error("Error running inference in graph.");
  }

  auto outputTensorValues = outputTensor.tensor<float, 2>();

  // Pass the results to the output output vector
  std::vector<std::vector<float> > batch_output;
  for (int i = 0; i < outputTensorValues.dimension(0); i++) {
    std::vector<float> output;
    for (int j = 0; j < outputTensorValues.dimension(1); j++) {
      output.push_back(outputTensorValues(i, j));
    }
    batch_output.push_back(output);
  }

  return batch_output;
}

void TensorflowGraphExecutor::batchFullForwardPass(
    const std::vector<Array3D>& inputs,
    const std::string& input_tensor_name,
    const std::vector<std::vector<float> >& scales,
    const std::string& scales_tensor_name,
    const std::string& descriptor_values_name,
    const std::string& reconstruction_values_name,
    std::vector<std::vector<float> >& descriptors,
    std::vector<Array3D>& reconstructions) const {
  CHECK(!inputs.empty());
  descriptors.clear();
  reconstructions.clear();

  // define the input placeholder
  auto inputShape = TensorShape();
  inputShape.AddDim((int64) inputs.size());
  inputShape.AddDim((int64) inputs[0].container.size());
  inputShape.AddDim((int64) inputs[0].container[0].size());
  inputShape.AddDim((int64) inputs[0].container[0][0].size());
  inputShape.AddDim((int64) 1u);
  Tensor inputTensor(DT_FLOAT, inputShape);

  // put values in the input tensor
  auto inputTensorValues = inputTensor.tensor<float, 5>();
  for (size_t i=0; i < inputs.size(); i++) {
    for (size_t j=0; j < inputs[0].container.size(); j++) {
      for (size_t k=0; k < inputs[0].container[0].size(); k++) {
        for (size_t l=0; l < inputs[0].container[0][0].size(); l++) {
          inputTensorValues(i, j, k, l, 0) = inputs[i].container[j][k][l];
        }
      }
    }
  }

  auto scales_shape = TensorShape();
  scales_shape.AddDim((int64) inputs.size());
  scales_shape.AddDim(3u);
  //scales_shape.AddDim(1u);
  Tensor scales_tensor(DT_FLOAT, scales_shape);

  // put values in the input tensor
  auto scales_tensor_values = scales_tensor.tensor<float, 2>();
  for (size_t i=0; i < scales.size(); i++) {
    for (size_t j=0; j < scales[0].size(); j++) {
      scales_tensor_values(i, j) = scales[i][j];
    }
  }

  std::vector<Tensor> output_tensors;
  Status status = this->executeGraph(
      {{input_tensor_name, inputTensor}, {scales_tensor_name, scales_tensor}},
      {descriptor_values_name, reconstruction_values_name},
      output_tensors);

  if (!status.ok()) {
      LOG(INFO) << status.error_message();
  }
  CHECK(status.ok());
  CHECK_EQ(output_tensors.size(), 2u);

  auto descriptor_values = output_tensors[0].tensor<float, 2>();
  auto reconstruction_values = output_tensors[1].tensor<float, 5>();

  CHECK_EQ(descriptor_values.dimension(0), reconstruction_values.dimension(0));
  CHECK_EQ(descriptor_values.dimension(0), inputs.size());

  const unsigned int batch_size =  inputs.size();
  std::vector<float> descriptor;
  Array3D reconstruction(inputs[0].container.size(),
                         inputs[0].container[0].size(),
                         inputs[0].container[0][0].size());
  for (unsigned int i = 0u; i < batch_size; ++i) {
    descriptor.clear();
    for (int j = 0; j < descriptor_values.dimension(1); j++) {
      descriptor.push_back(descriptor_values(i, j));
    }

    for (size_t j=0; j < inputs[0].container.size(); j++) {
      for (size_t k=0; k < inputs[0].container[0].size(); k++) {
        for (size_t l=0; l < inputs[0].container[0][0].size(); l++) {
          reconstruction.container[j][k][l] = reconstruction_values(i, j, k, l, 0);
        }
      }
    }

    descriptors.push_back(descriptor);
    reconstructions.push_back(reconstruction);
  }
}

tensorflow::Status TensorflowGraphExecutor::executeGraph(
    const Tensor& inputTensor, Tensor& outputTensor, const std::string& input_tensor_name,
    const std::string& output_tensor_name) const {
  // Run the session, evaluating the outputOpName operation from the graph
  vector<Tensor> outputTensors;

  auto status = this->executeGraph(
      {{input_tensor_name, inputTensor}},
      {output_tensor_name},
      outputTensors);

  if (status.ok()) {
    outputTensor = outputTensors[0];
  }

  return status;
}

tensorflow::Status
TensorflowGraphExecutor::executeGraph(const std::vector<std::pair<std::string, tensorflow::Tensor>>& feedDict,
                                      const std::vector<std::string>& outputOps,
                                      std::vector<tensorflow::Tensor>& outputTensors) const {
  // Run the session, evaluating the outputOpName operation from the graph
  return this->tensorflowSession->Run(feedDict, outputOps, {}, &outputTensors);
}

TensorflowGraphExecutor::~TensorflowGraphExecutor() {
  this->tensorflowSession->Close();

  delete this->tensorflowSession;
  delete this->graph_def;
}

};
