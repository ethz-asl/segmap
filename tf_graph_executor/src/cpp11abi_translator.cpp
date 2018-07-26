#define _GLIBCXX_USE_CXX11_ABI 0

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

using namespace tensorflow;

namespace tf_graph_executor {

Status readBinaryProto(Env* env, const char* fname,
                       ::tensorflow::protobuf::MessageLite* proto) {
  LOG(INFO) << "In custom function";
  return tensorflow::ReadBinaryProto(env, fname, proto);
}

}
