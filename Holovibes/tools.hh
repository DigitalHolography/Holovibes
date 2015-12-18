#pragma once

# include <cufft.h>

template<typename Container, typename Functor>
void delete_them(Container& c, const Functor& f)
{
  std::for_each(c.begin(),
    c.end(),
    f);
  c.clear();
}

/*! Converting cartesian data to polar data.
* \param data An array of complex values (each being a struct
* of two floats). The module will be stored first, then the angle.
*/
void to_polar(cufftComplex* data, const size_t size);