/*! \file tools.hh
 *
 * Generic, widely usable functions. */
#pragma once

# include <algorithm>
# include <cmath>

# include "hardware_limits.hh"

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