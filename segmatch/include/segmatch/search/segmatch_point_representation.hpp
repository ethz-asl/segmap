#ifndef SEGMATCH_POINT_REPRESENTATION_H_
#define SEGMATCH_POINT_REPRESENTATION_H_

#include <segmatch/common.hpp>
#include <segmatch/point_extended.hpp>
#include <pcl/point_representation.h>

namespace pcl {
template <>
class DefaultPointRepresentation <segmatch::PointExtended> : public  PointRepresentation <segmatch::PointExtended> {
 public:
   DefaultPointRepresentation () {
     nr_dimensions_ = 5;
     trivial_ = false;
   }

   virtual void
   copyToFloatArray (const segmatch::PointExtended &p, float * out) const {
     out[0] = p.x;
     out[1] = p.y;
     out[2] = p.z;
     // Semantic class
     out[3] = p.a;
     // Color information
     uint32_t rgb = p.rgba & 0xffffff;
     out[4] = segmatch::rgb_to_hue(rgb);
   }
};
}

#endif  // SEGMATCH_POINT_REPRESENTATION_H_
