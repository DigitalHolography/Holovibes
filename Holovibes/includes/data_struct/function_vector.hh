/*! \file
 *
 * \brief Define FunctionVector class.
 */

#pragma once
#include "aliases.hh"
namespace holovibes
{
/*! \class FunctionVector
 *
 *  \brief Reprensent the main vector of the pipe. Used to store all functions in the pipe and run all of them only when
 *  needed.
 *  This class is a wrapper of std::vector class from the STL.
 *  The type of the vector is std::vector<std::pair<ushort, FnType>>
 *  Each functions is associated to a unique id, allowing to retrieve them when needed. e.g: when we need to
 *  remove a function.
 */
class FunctionVector
{
  public:
    /*! \brief Constuctor */
    FunctionVector() = default;

    /*! \brief Copy constructor. */
    FunctionVector(const FunctionVector& other)
        : fn_vect_(other.fn_vect_)
        , next_id_(other.next_id_.load())
    {
    }

    ~FunctionVector() = default;

    /*! \brief Assignment operator (Maybe store fnvect in the pipe in a make shared in the future). */
    FunctionVector& operator=(const FunctionVector& other)
    {
        if (this != &other)
        {
            fn_vect_ = other.fn_vect_;
            next_id_ = other.next_id_.load();
        }
        return *this;
    }

    /*! \brief Execute all the functions in the `fn_vect_` vector.
     *  If we need to remove some functions from the vector at the end of the execution, their IDs are taken from
     *  `remove_vect_` and then erased from `fn_vect_`.
     */
    void call_all();

    /*! \brief Stop the execution */
    void exit_now() { exit_ = true; }

    /*! \brief Push back the function in the vector. Get a new unique ID associated to the function.
     *
     *  \param[in] function The reference to the function to push.
     *
     *  \return The id of the function in the vector.
     */
    ushort push_back(const FnType& function);

    /*! \brief Remove a function of the vector by its ID.
     *  Function is removed at the end of `fn_vect_` execution.
     *
     *  \param[in] id The unique id of the function.
     */
    void remove(const ushort id);

    /*! \brief Remove a function of the vector by its ID. The function is only when `remove_condition` is true.
     *  Function is removed at the end of `fn_vect_` execution.
     *
     *  \param[in] id The unique id of the function.
     *  \param[in] remove_condition The condition waited before removing the function from the vector.
     */
    void conditionnal_remove(const ushort id, const ConditionType& remove_condition);

    /*! \brief Clear all the vector and reset the IDs. */
    void clear();

  protected:
    /*! \brief Erase a given function in the vector by its ID.
     *  Calls `erase` function from `fn_vect_`.
     *
     *  \param[in] id The id of the function to remove.
     */
    void erase(const ushort id);

  private:
    /*! \brief The ID generator for the unique IDs of the functions. Reset to 0 when `clear` is called. */
    std::atomic<ushort> next_id_;

    /*! \brief The vector used to store the pair of id and functions. */
    FnVector fn_vect_;

    /*! \brief The vector used to store the functions to remove from the `fn_vect_`. Cleared at the end of `call_all`.
     */
    std::vector<ushort> remove_vect_;

    /*! \brief Tells whether to run or exit */
    bool exit_;
};
} // namespace holovibes