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

/*! \file
 *
 * Utility functions and types used in ICompute-based classes. */
#pragma once

# include "queue.hh"
# include <deque>

namespace holovibes
{
  //!< A single procedure.
  using FnType = std::function<void()>;
  //!< A procedure vector.
  using FnVector = std::vector<FnType>;
  //!< A procedure deque.
  using FnDeque = std::deque<FnType>;
}

bool queue_enqueue(void* input, holovibes::Queue* queue, holovibes::SquareInputMode mode = holovibes::SquareInputMode::NO_MODIFICATION);