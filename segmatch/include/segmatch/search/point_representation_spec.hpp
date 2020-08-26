#ifndef SEGMATCH_POINT_REPRESENTATION_SPEC_H_
#define SEGMATCH_POINT_REPRESENTATION_SPEC_H_

#include <algorithm>

#include <pcl/point_representation.h>

#include "segmatch/common.hpp"


namespace pcl {

// derive DefaultPointRepresentation class
// TODO: maybe create generic base class and create specialization for color and/or color + semantics
template <typename PointDefault>
  class SemanticPointRepresentation : public PointRepresentation<PointDefault>
  {
    using PointRepresentation<PointDefault>::nr_dimensions_;
    using PointRepresentation<PointDefault>::trivial_;
    // using PointRepresentation<PointDefault>::makeShared;

    public:
      // Boost shared pointers
      typedef boost::shared_ptr<SemanticPointRepresentation<PointDefault> > Ptr;
      typedef boost::shared_ptr<const SemanticPointRepresentation<PointDefault> > ConstPtr;

      SemanticPointRepresentation()
      {
        nr_dimensions_ = 7;
        trivial_ = true;
      }

      virtual ~SemanticPointRepresentation () {}


      inline Ptr
      makeShared () const
      {
        return (Ptr (new SemanticPointRepresentation<PointDefault> (*this)));
      }

      virtual void
      copyToFloatArray (const PointDefault &p, float * out) const
      {
        // If point type is unknown, treat it as a struct/array of floats
        const float* ptr = reinterpret_cast<const float*> (&p);
        for (int i = 0; i < nr_dimensions_; ++i)
          out[i] = ptr[i];
      }
  };

  // TODO: move this function somewhere else
  static float rgb_to_hue(uint32_t r, uint32_t g, uint32_t b)
  {
    std::cout << "Converting rgb to hue!" << std::endl;
    float hue = 0;
    std::vector<float> rgb;
    rgb.push_back(r / 255.0f);
    rgb.push_back(g / 255.0f);
    rgb.push_back(b / 255.0f);

    uint8_t max_index = std::distance(rgb.begin(), std::max_element(rgb.begin(), rgb.end()));
    uint8_t min_index = std::distance(rgb.begin(), std::min_element(rgb.begin(), rgb.end()));

    float diff = (rgb[max_index] - rgb[min_index]);

    // avoid overflow error
    if (diff < 0.000001) {
        return 0;
    }

    // if red has the max value
    if (0 == max_index) {
        hue = (rgb[1] - rgb[2]) / diff * 60;
    }
    else if (1 == max_index) {
        hue = (2.0 + (rgb[2] - rgb[0]) / diff) * 60;
    }
    else if (2 == max_index) {
        hue = (4.0 + (rgb[0] - rgb[1]) / diff) * 60;
    }

    return hue >= 0 ? hue : hue + 360;

  }

  // TODO: this class is for 4D color points, add one for 5D points (color + semantics) and 4D semantics points
  template<>
  class SemanticPointRepresentation<segmatch::MapPoint> : public PointRepresentation<segmatch::MapPoint>
  {
    public:
      SemanticPointRepresentation()
      {
          nr_dimensions_ = 4;
          trivial_ = true;
      }

      // TODO: potentially add functionality to convert rgb to hue here since that's all we need for segmentation.
      virtual void copyToFloatArray(const segmatch::MapPoint& p, float* out) const
      {
          out[0] = p.x;
          out[1] = p.y;
          out[2] = p.z;

          // use hue instead of color
          // float hue = rgb_to_hue(p.r, p.g, p.b);
          // out[3] = hue;
          // uint32_t semantic_rgb = ((uint32_t)p.semantics_r << 16 | (uint32_t)p.semantics_g << 8 | (uint32_t)p.semantics_b);
          // uint32_t semantic_rgb = ((uint32_t)111 << 16 | (uint32_t)112 << 8 | (uint32_t)113);
          // out[3] = static_cast<float>(semantic_rgb);
          out[3] = static_cast<float>(p.semantics_rgb);
        //   out[3] = *reinterpret_cast<float*>(&semantic_rgb);

          // p.semantics_rgb is equal to the constructor default value here if the points I'm reading (or the map) don't
          // contain the field 'semantics_rgb'
        //   std::cout << "in copyToFloatArray semantic_rgb:" << std::to_string(p.semantics_rgb) << std::endl;
        //   std::cout << "in copyToFloatArray rgb:" << std::to_string(p.rgb) << std::endl;
        //   std::cout << "in copyToFloatArray out[3]:" << std::to_string(out[3]) << " or " << out[3] << " or "
        //             << std::to_string(semantic_rgb) << std::endl;
          //   std::cout << "in copyToFloatArray:" << std::to_string(p.r) << ", " << std::to_string(p.g) << ", "
          //             << std::to_string(p.b) << std::endl;
        //   std::cout << "out[3] = " << std::to_string(out[3]) << " which should equal " << std::to_string(semantic_rgb) << std::endl;
      }

      void printNumDims() const { std::cout << "num dims from inside class: " << nr_dimensions_ << std::endl; }

      static constexpr int existance = 447;
      int another_test = 43;
  };
  } // namespace pcl

#endif // SEGMATCH_POINT_REPRESENTATION_SPEC_H_
