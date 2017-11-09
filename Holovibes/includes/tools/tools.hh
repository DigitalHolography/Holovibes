/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file tools.hh
 *
 * Generic, widely usable functions. */
#pragma once

# include <algorithm>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
# include <cmath>
# include <math.h>
# include <string>
# include <ctime>
# include <qrect.h>

# include "units/rect.hh"
# include "hardware_limits.hh"
# include "frame_desc.hh"
# include "cufft.h"

using namespace holovibes;

/*! \function Generic loop for deleting a container's elements. */
template<typename Container, typename Functor>
void delete_them(Container& c, const Functor& f)
{
  std::for_each(c.begin(), c.end(), f);
  c.clear();
}

/*! \function Given a problem of *size* elements, compute the lowest number of
 * blocks needed to fill a compute grid.
 *
 * \param nb_threads Number of threads per block. */
inline unsigned map_blocks_to_problem(const size_t problem_size,
  const unsigned nb_threads)
{
  unsigned nb_blocks = static_cast<unsigned>(
    std::ceil(static_cast<float>(problem_size) / static_cast<float>(nb_threads)));

  if (nb_blocks > get_max_blocks())
    nb_blocks = get_max_blocks();

  return nb_blocks;
}


/*! \brief cast a framedescriptor into a Rectangle
inline units::RectWindow get_rectangle(const camera::FrameDescriptor& a)
{
	return units::RectWindow(0, 0, a.width, a.height);
}
 */

inline double clockToMilliseconds(clock_t ticks)
{
	// units/(units/time) => time (seconds) * 1000 = milliseconds
	return (ticks / static_cast<double>(CLOCKS_PER_SEC)) * 1000.0;
}


namespace holovibes
{
	/*! \brief Get framerate from .cine file */
	void	get_framerate_cinefile(FILE *file, std::string &file_src_);
	/*! \brief Get exposure from .cine file */
	void	get_exposure_cinefile(FILE *file, std::string &file_src_);
	/*! \brief Calculate the nearest upper power of 2 */
	unsigned short	upper_window_size(ushort width, ushort height);
	/*! \brief Prints a float buffer allocated on gpu*/
	void print_gpu_buffer(const float* buf, std::size_t nb_elts);
	/*! \brief Prints a complex buffer allocated on gpu*/
	void print_gpu_buffer(const cuComplex* buf, std::size_t nb_elts);
}