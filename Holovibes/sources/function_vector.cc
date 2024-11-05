#include "function_vector.hh"
namespace holovibes
{
FunctionVector::FunctionVector(ConditionType condition)
    : condition_(condition)
    , next_id_(0)
{
}

void FunctionVector::call_all()
{
    // Call all functions in the vector.
    for (const auto& pair : fn_vect_)
    {
        const FnType& f = pair.second;
        f();
    }

    // If some functions need to be removed, remove them.
    for (const auto& elt_id : remove_vect_)
    {
        erase(elt_id);
    }
    remove_vect_.clear();
}

int FunctionVector::push_back(FnType function)
{
    // Get a new unique ID for the function to push.
    int id = next_id_++;
    fn_vect_.push_back({id, function});
    return id;
}

int FunctionVector::conditional_push_back(const FnType& function)
{
    // Get a new unique ID for the function to push.
    int id = next_id_++;
    fn_vect_.push_back({id,
                        [=]()
                        {
                            if (!condition_())
                                return;
                            function();
                        }});
    return id;
}

void FunctionVector::conditionnal_remove(int id, ConditionType remove_condition)
{
    int remove_id = next_id_++;
    FnType wrapped_function = [=]()
    {
        if (remove_condition())
        {
            this->remove_vect_.push_back(id);
            // Because we push this function in the `fn_vect_`, we also need to remove it since we do not need it
            // anymore.
            this->remove_vect_.push_back(remove_id);
        }
    };
    fn_vect_.push_back({id, wrapped_function});
}

void FunctionVector::erase(int id)
{
    auto it = std::find_if(fn_vect_.begin(), fn_vect_.end(), [id](const auto& pair) { return pair.first == id; });
    if (it != fn_vect_.end())
        fn_vect_.erase(it);
}

} // namespace holovibes