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
    exit = false;
    for (const auto& [_, f] : fn_vect_)
    {
        f();
        if (exit)
            break;
    }

    // If some functions need to be removed, remove them.
    for (const auto& elt_id : remove_vect_)
        erase(elt_id);

    remove_vect_.clear();
}

ushort FunctionVector::push_back(const FnType& function)
{
    // Get a new unique ID for the function to push.
    ushort id = next_id_++;
    fn_vect_.push_back({id, function});
    return id;
}

ushort FunctionVector::conditional_push_back(const FnType& function)
{
    // Get a new unique ID for the function to push.
    ushort id = next_id_++;
    fn_vect_.push_back({id,
                        [=]()
                        {
                            // if (!condition_())
                            //     return;
                            function();
                        }});
    return id;
}

void FunctionVector::remove(const ushort id) { this->remove_vect_.push_back(id); }

void FunctionVector::conditionnal_remove(const ushort id, const ConditionType& remove_condition)
{
    ushort remove_id = next_id_++;
    fn_vect_.push_back({id,
                        [=]()
                        {
                            if (remove_condition())
                            {
                                this->remove_vect_.push_back(id);
                                // Because we push this function in the `fn_vect_`, we also need to remove it since we
                                // do not need it anymore.
                                this->remove_vect_.push_back(remove_id);
                            }
                        }});
}

void FunctionVector::erase(const ushort id)
{
    auto it = std::find_if(fn_vect_.begin(), fn_vect_.end(), [id](const auto& pair) { return pair.first == id; });
    if (it != fn_vect_.end())
        fn_vect_.erase(it);
}

void FunctionVector::clear()
{
    fn_vect_.clear();
    remove_vect_.clear();
    next_id_ = 0;
}

} // namespace holovibes