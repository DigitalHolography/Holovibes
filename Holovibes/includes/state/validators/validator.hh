#pragma once

#include <stdexcept>

namespace holovibes::validators
{

/*!
 * \brief Default exception launched by validators
 * 
 */
struct ValidationException : std::invalid_argument
{
};

/*!
 * \brief The inheritance of this class are validators, they poses constraints on the input in this placeholder.
 * 
 * @tparam T the object stored
 */
template <typename T>
struct Validator
{
    /*!
     * \brief Construct a new Validator object and store it as a proxy
     * 
     * @param obj 
     */
    Validator(T obj)
        : obj_(obj)
    {
        if (!validate(obj))
            raise error();
    }

    /*!
     * \brief Member to overload to create new validators
     * 
     * @param value 
     * @return true passes the validation
     * @return false rejected, launches an exception (specified by the error() function)
     */
    virtual bool validate(T value) const noexcept = 0;

    /*!
     * \brief Return the exception to be raised if the validation failed
     * 
     * @return std::exception return type can be any type of exception, default is ValidationException
     */
    virtual std::exception error() { return ValidationException("Constraints are not respected"); }

    /*!
     * \brief Object stored should be easily extractable
     * 
     * @return the object stored
     */
    T operator T() const { return obj_; }

  private:
    const T obj_;
};

} // namespace holovibes::validators