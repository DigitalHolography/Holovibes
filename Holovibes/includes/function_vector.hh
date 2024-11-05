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
class FunctionVector
{
  public:
    FunctionVector(ConditionType condition);
    FunctionVector() = default;
    ~FunctionVector() = default;
    FunctionVector(const FunctionVector& other)
        : condition_(other.condition_)
        , fn_vect_(other.fn_vect_)
        , next_id_(other.next_id_.load())
    {
    }
    // Assignment operator (Maybe store fnvect in a make shared in the future)
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
    /*! \brief Push back the function in the vector.
     *
     *  \param[in] function The function to push.
     *
     *  \return The id of the function in the vector.
     */
    int push_back(FnType function);
    /*! \brief Push back the function in the vector depending on the condition.
     *
     * Execute it only if the condition is verified.
     * in the pipe, the condition is set in the constructor, and is the following :
     *
     * ConditionType batch_condition = [&]() -> bool
     * { return batch_env_.batch_index == setting<settings::TimeStride>(); };
     *
     *  \param[in] function The reference to the function to push.
     */
    void conditional_push_back(const FnType& function);
    /*! \brief Remove a given function in the vector.
     *
     *  \param[in] id The id of the function to remove.
     */
    void remove(int id);
    void call_all();
    // void conditional_push_back_remove(const FnType& function);
    // Efface toutes les fonctions
    void clear() { fn_vect_.clear(); }
    void conditional_push_back_remove(const FnType& function, ConditionType condition);

  private:
    ConditionType condition_;
    std::atomic<int> next_id_;
    FnVector fn_vect_;
    std::vector<int> remove_vect_;
    bool end_call_all = false;
};
} // namespace holovibes