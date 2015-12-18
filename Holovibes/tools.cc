#include <cmath>

#include "tools.hh"

void to_polar(cufftComplex* data, const size_t size)
{
  for (auto i = 0; i < size; ++i)
  {
    float dist = std::hypotf(data[i].x, data[i].y);
    float angle = std::atan(data[i].y / data[i].x);
    data[i].x = dist;
    data[i].y = angle;
  }
}