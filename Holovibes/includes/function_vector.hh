/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "aliases.hh"

namespace holovibes
{
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
     * in the pipe, the condition is set in the constructor, and is the following :
     * 
     * ConditionType batch_condition = [&]() -> bool
     * { return batch_env_.batch_index == setting<settings::TimeStride>(); };
     */
    void conditional_push_back(const FnType& function);

  private:
    ConditionType condition_;
};
} // namespace holovibes
