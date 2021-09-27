/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "pipeline_utils.hh"

namespace holovibes
{
using ConditionType = std::function<bool()>;

/*! \class FunctionVector
 *
 * \brief #TODO Add a description for this class
 */
class FunctionVector : public FnVector
{
  public:
    FunctionVector(ConditionType condition);

    FunctionVector() = default;

    ~FunctionVector() = default;

    /*! \brief Push back the function in the vector.
     *
     * Execute it only if the condition is verified.
     */
    void conditional_push_back(const FnType& function);

  private:
    ConditionType condition_;
};
} // namespace holovibes