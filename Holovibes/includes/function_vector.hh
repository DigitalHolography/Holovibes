/*! \file
 *
 * \brief Defines the FunctionVector class for managing conditional function execution.
 */
#pragma once

#include "pipeline_utils.hh"
#include <functional>

namespace holovibes
{
using ConditionType = std::function<bool()>;

/*! \class FunctionVector
 *
 * \brief Manages a vector of functions that are executed conditionally.
 *
 * This class extends FnVector to include a condition that must be met
 * for functions to be added to the vector.
 */
class FunctionVector : public FnVector
{
  public:
    /*! \brief Constructor with a condition
     *
     * \param condition Condition that must be satisfied for functions to be added.
     */
    FunctionVector(ConditionType condition);

    /*! \brief Default constructor */
    FunctionVector() = default;

    /*! \brief Default destructor */
    ~FunctionVector() = default;

    /*! \brief Adds a function to the vector if the condition is satisfied
     *
     * \param function Function to be added.
     */
    void conditional_push_back(const FnType& function);

  private:
    ConditionType condition_; /*!< Condition for adding functions to the vector */
};
} // namespace holovibes