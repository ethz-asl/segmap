#include "segmatch/common.hpp"

namespace segmatch {

double rgb_to_hue(uint32_t rgb) {
  float r = ((rgb >> 16) & 0xff) / 255.0;
  float g = ((rgb >> 8) & 0xff) / 255.0;
  float b = (rgb & 0xff) / 255.0;

  float max = std::max(r, std::max(g, b));
  float min = std::min(r, std::min(g, b));
  float diff = max - min;

  // avoid overflow
  if (diff < 1e-6) {
    return 0.0;
  }

  float hue = 0.0;
  if (abs(max - r) < 1e-6) {
    hue = (g - b) / diff;
  } else if (abs(max - g) < 1e-6) {
    hue = 2.0 + (b - r) / diff;
  } else if (abs(max - b) < 1e-6) {
    hue = 4.0 + (r - g) / diff;
  }

  hue *= 60.0;
  if (hue < 0) {
    hue += 360.0;
  }
  hue /= 360.0;

  return hue;
}

}
