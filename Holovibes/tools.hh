/*! \file tools.hh
 *
 * Generic, widely usable functions. */
#pragma once

# include <algorithm>
# include <cmath>
# include <string>

# include "geometry.hh"
# include "hardware_limits.hh"
# include "frame_desc.hh"

/*! \function Generic loop for deleting a container's elements. */
template<typename Container, typename Functor>
void delete_them(Container& c, const Functor& f)
{
  std::for_each(c.begin(),
    c.end(),
    f);
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


/*! \brief cast a framedescriptor into a Rectangle */
inline holovibes::Rectangle get_rectangle(const camera::FrameDescriptor& a)
{
	return holovibes::Rectangle(a.width, a.height);
}


namespace holovibes
{
	/*! \brief Get framerate from .cine file */
	void	get_framerate_cinefile(FILE *file, std::string &file_src_);
	/*! \brief Get exposure from .cine file */
	void	get_exposure_cinefile(FILE *file, std::string &file_src_);
	/*! \brief Calculate the nearest squared window */
	unsigned short	nearest_window_size(const camera::FrameDescriptor frame);
	/*! \brief Calculate the nearest power of two */
	unsigned short	nearest_size(const unsigned short n);
	/*! \brief Cast buffer into real_buffer*/
	void	buffer_size_conversion(char *real_buffer
		, const char *buffer
		, const camera::FrameDescriptor real_frame_desc
		, const camera::FrameDescriptor frame_desc);
}