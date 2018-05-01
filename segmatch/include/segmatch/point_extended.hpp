#ifndef SEGMATCH_POINT_EXTENDED_HPP_
#define SEGMATCH_POINT_EXTENDED_HPP_

#include <stdint.h>

#define PCL_NO_PRECOMPILE
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/impl/instantiate.hpp>

/// \brief Custom PCL point type that can be extended with custom fields needed by incremental
/// algorithms. Custom points must be defined and registered in the global namespace.
struct _SegMatch_PointExtended {
  inline _SegMatch_PointExtended(const _SegMatch_PointExtended &p)
    : data { p.x, p.y, p.z, 1.0f }, ed_cluster_id(p.ed_cluster_id),
      sc_cluster_id(p.sc_cluster_id) {
  }

  inline _SegMatch_PointExtended()
    : data { 0.0f, 0.0f, 0.0f, 1.0f }, ed_cluster_id(0u), sc_cluster_id(0u) {
  }

  friend std::ostream& operator << (std::ostream& os, const _SegMatch_PointExtended& p) {
    return os << "x: "<< p.x << ", y: " << p.y << ", z: " << p.z
        << ", EuclideanDistance ID: " << p.ed_cluster_id
        << ", SmoothnessConstraints ID: " << p.sc_cluster_id;
  }

  // X, Y, Z components of the position of the point.
  // Memory layout (4 x 4 bytes): [ x, y, z, _ ]
  PCL_ADD_POINT4D;
  // Cluster ID fields.
  // Memory layout (4 x 4 bytes): [ ed_cluster_id, sc_cluster_id, _, _ ]
  union {
    struct {
      uint32_t ed_cluster_id;
      uint32_t sc_cluster_id;
    };
    uint32_t data_c[4];
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Register the point type.
POINT_CLOUD_REGISTER_POINT_STRUCT (_SegMatch_PointExtended,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (uint32_t, ed_cluster_id, ed_cluster_id)
                                   (uint32_t, sc_cluster_id, sc_cluster_id)
)

namespace segmatch {
  typedef _SegMatch_PointExtended PointExtended;
} // namespace segmatch

#endif // SEGMATCH_POINT_EXTENDED_HPP_
