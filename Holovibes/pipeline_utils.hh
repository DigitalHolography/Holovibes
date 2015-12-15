#pragma once

# include <functional>
# include <vector>
# include <deque>

namespace holovibes
{
  /*! \brief Vector of procedures type */
  using FnType = std::function<void()>;
  using FnVector = std::vector<FnType>;
  using FnDeque = std::deque<FnType>;
}