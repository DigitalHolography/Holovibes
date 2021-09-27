/*! \file
 *
 * \brief Utility functions and types used in ICompute-based classes.
 */
#pragma once

#include <vector>

namespace holovibes
{
/*! \brief A single procedure. */
using FnType = std::function<void()>;
/*! \brief A procedure vector. */
using FnVector = std::vector<FnType>;
/*! \brief A procedure deque. */
using FnDeque = std::deque<FnType>;
} // namespace holovibes
