/*! \file
 *
 * Utility functions and types used in ICompute-based classes. */
#pragma once

#include <vector>

namespace holovibes
{
//!< A single procedure.
using FnType = std::function<void()>;
//!< A procedure vector.
using FnVector = std::vector<FnType>;
//!< A procedure deque.
using FnDeque = std::deque<FnType>;
} // namespace holovibes
