#pragma once

# include <functional>
# include <vector>

namespace holovibes
{
  /*! \brief Vector of procedures type */
  using FnType = std::function<void()>;
  using FnVector = std::vector<FnType>;
}