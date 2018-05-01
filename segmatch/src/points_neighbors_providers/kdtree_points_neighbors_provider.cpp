#include "segmatch/points_neighbors_providers/impl/kdtree_points_neighbors_provider.hpp"

namespace segmatch {
// Instantiate KdTreePointsNeighborsProvider for the template parameters used in the
// application.
template class KdTreePointsNeighborsProvider<MapPoint>;
// Add any other required instantiation here or in a separate file and declare them in
// segmatch/points_neighbors_providers/impl/kdtree_points_neighbors_provider.hpp.
} // namespace segmatch
