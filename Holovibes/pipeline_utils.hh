#pragma once

# include <functional>
# include <vector>

namespace holovibes
{
  /*! \brief Vector of procedures type */
  using FnVector = std::vector < std::function<void()> >;
}
