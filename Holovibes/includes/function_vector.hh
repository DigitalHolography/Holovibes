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
    /*! \brief Constuctor
     *  \param[in] condition The condition used in `conditional_push_back` function.
     */
    FunctionVector(ConditionType condition);
    FunctionVector() = default;

    /*! \brief Copy constructor. */
    FunctionVector(const FunctionVector& other)
        : condition_(other.condition_)
        , fn_vect_(other.fn_vect_)
        , next_id_(other.next_id_.load())
    {
    }

    ~FunctionVector() = default;

    /*! \brief Assignment operator (Maybe store fnvect in the pipe in a make shared in the future). */
    FunctionVector& operator=(const FunctionVector& other)
    {
        if (this != &other)
        {
            condition_ = other.condition_;
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

    /*! \brief Push back the function in the vector. Get a new unique ID associated to the function.
     *
     *  \param[in] function The function to push.
     *
     *  \return The id of the function in the vector.
     */
    ushort push_back(FnType function);

    /*! \brief Push back the function in the vector depending on the condition.
     *
     *  Execute it only if the condition is verified.
     *  in the pipe, the condition is set in the constructor, and is the following :
     *
     *  ConditionType batch_condition = [&]() -> bool
     *  { return batch_env_.batch_index == setting<settings::TimeStride>(); };
     *
     *  \param[in] function The reference to the function to push.
     *
     *  \return The id of the function in the vector.
     */
    ushort conditional_push_back(const FnType& function);

    /*! \brief Remove a function of the vector by its ID.
     *  Function is removed at the end of `fn_vect_` execution.
     *
     *  \param[in] id The unique id of the function.
     */
    void remove(ushort id);

    /*! \brief Remove a function of the vector by its ID. The function is only when `remove_condition` is true.
     *  Function is removed at the end of `fn_vect_` execution.
     *
     *  \param[in] id The unique id of the function.
     *  \param[in] remove_condition The condition waited before removing the function from the vector.
     */
    void conditionnal_remove(ushort id, ConditionType remove_condition);

    /*! \brief Erase a given function in the vector by its ID.
     *  Calls `erase` function from `fn_vect_`.
     *
     *  \param[in] id The id of the function to remove.
     */
    void erase(ushort id);

    /*! \brief Clear all the vector and reset the IDs. */
    void clear();

  private:
    /*! \brief The condition used in `conditional_push_back` */
    ConditionType condition_;

    /*! \brief The ID generator for the unique IDs of the functions. Reset to 0 when `clear` is called. */
    std::atomic<ushort> next_id_;

    /*! \brief The vector used to store the pair of id and functions. */
    FnVector fn_vect_;

    /*! \brief The vector used to store the functions to remove from the `fn_vect_`. Cleared at the end of `call_all`.
     */
    std::vector<ushort> remove_vect_;
};
} // namespace holovibes