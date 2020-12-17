/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
