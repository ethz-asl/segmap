// This code was originally written by Martin Pecka ( martin.pecka@cvut.cz ) and adapted
// for our application.

#ifndef TF_GRAPH_EXECUTOR_TF_GRAPH_EXECUTOR_HPP
#define TF_GRAPH_EXECUTOR_TF_GRAPH_EXECUTOR_HPP

#include <string>
#include <vector>

// We need to use tensorflow::* classes as PIMPL
namespace tensorflow {
class Session;
class Status;
class Tensor;
class MetaGraphDef;
};

namespace tf_graph_executor {

struct Array3D {
  Array3D(unsigned int x_dim, unsigned int y_dim, unsigned int z_dim) {
    resize(x_dim, y_dim, z_dim);
    init();
  }

  void resize(unsigned int x_dim, unsigned int y_dim, unsigned int z_dim) {
    container.resize(x_dim);
    for (auto& y : container) {
      y.resize(y_dim);
      for (auto& z : y) {
        z.resize(z_dim);
      }
    }
  }

  void init() {
    for (auto& x : container) {
      for (auto& y : x) {
        for (auto& z : y) {
          z = 0.0;
        }
      }
    }
  }

  std::vector<std::vector<std::vector<double> > > container;
};


class TensorflowGraphExecutor {
public:
    explicit TensorflowGraphExecutor(const std::string& pathToGraph);

    virtual ~TensorflowGraphExecutor();

    void loadCheckpoint(const std::string& checkpointPath);

    std::vector<float> executeGraph(const std::vector<float>& inputs,
                                    const std::string& input_tensor_name,
                                    const std::string& output_tensor_name) const;

    std::vector<std::vector<float> > batchExecuteGraph(
        const std::vector<std::vector<float> >& inputs, const std::string& input_tensor_name,
        const std::string& output_tensor_name) const;

    std::vector<std::vector<float> > batchExecuteGraph(
        const std::vector<Array3D>& inputs, const std::string& input_tensor_name,
        const std::string& output_tensor_name) const;

    void batchFullForwardPass(
        const std::vector<Array3D>& inputs,
        const std::string& input_tensor_name,
        const std::vector<std::vector<float> >& scales,
        const std::string& scales_tensor_name,
        const std::string& descriptor_tensor_name,
        const std::string& reconstruction_tensor_name,
        std::vector<std::vector<float> >& descriptors,
        std::vector<Array3D>& reconstructions) const;

    tensorflow::Status executeGraph(const tensorflow::Tensor& inputTensor,
                                    tensorflow::Tensor& outputTensor,
                                    const std::string& input_tensor_name,
                                    const std::string& output_tensor_name) const;

    tensorflow::Status executeGraph(const std::vector<std::pair<std::string, tensorflow::Tensor> >& feedDict,
                                    const std::vector<std::string>& outputOps,
                                    std::vector<tensorflow::Tensor>& outputTensors) const;

protected:
    tensorflow::Session* tensorflowSession;

    tensorflow::MetaGraphDef* graph_def;
};

}


#endif //TF_GRAPH_EXECUTOR_TF_GRAPH_EXECUTOR_HPP
