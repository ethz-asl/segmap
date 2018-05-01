#include "segmatch/points_neighbors_providers/impl/octree_points_neighbors_provider.hpp"

namespace segmatch {
// Instantiate OctreePointsNeighborsProvider for the template parameters used in the
// application.
template class OctreePointsNeighborsProvider<MapPoint>;
// Add any other required instantiation here or in a separate file and declare them in
// segmatch/points_neighbors_providers/impl/octree_points_neighbors_provider.hpp.
} // namespace segmatch
