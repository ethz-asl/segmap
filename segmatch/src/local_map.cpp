#include "segmatch/impl/local_map.hpp"

#include "segmatch/common.hpp"

namespace segmatch {
// Instantiate LocalMap for the template parameters used in the application.
template class LocalMap<PclPoint, MapPoint>;
// Add any other required instantiation here or in a separate file and declare them in
// segmatch/impl/local_map.hpp.
} // namespace segmatch
