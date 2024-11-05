#include "function_vector.hh"
namespace holovibes
{
FunctionVector::FunctionVector(ConditionType condition)
    : condition_(condition)
    , next_id_(0)
{
}
// void FunctionVector::conditional_push_back(const FnType& function)
// {
//     push_back(
//         [=]()
//         {
//             if (!condition_())
//                 return;
//             function();
//         });
// }
// void FunctionVector::remove(const FnType& function)
// {
//     auto it = std::find(begin(), end(), function);
//     if (it != end())
//         erase(it);
// }
void FunctionVector::call_all()
{
    for (const auto& pair : fn_vect_)
    {
        const FnType& f = pair.second; // Récupérer la fonction
        f();                           // Appeler la fonction
    }
    for (const auto& elt_id : remove_vect_)
    {
        remove(elt_id);
    }
    remove_vect_.clear();
}
int FunctionVector::push_back(FnType function)
{
    int id = next_id_++;
    fn_vect_.push_back({id, function});
    return id;
}
void FunctionVector::conditional_push_back(const FnType& function)
{
    int id = next_id_++;
    FnType wrapped_function = [this, id, function]()
    {
        if (!condition_())
            return;
        function(); // Exécute la fonction
    };
    // Ajoute la fonction avec son ID
    fn_vect_.push_back({id, wrapped_function});
}
void FunctionVector::remove(int id)
{
    auto it = std::find_if(fn_vect_.begin(), fn_vect_.end(), [id](const auto& pair) { return pair.first == id; });
    if (it != fn_vect_.end())
        fn_vect_.erase(it);
}

void FunctionVector::conditional_push_back_remove(const FnType& function, ConditionType remove_condition)
{
    int id = next_id_++;
    FnType wrapped_function = [this, id, function, remove_condition]()
    {
        if (!condition_())
            return;
        function(); // Exécute la fonction
        if (remove_condition())
            this->remove_vect_.push_back(id);
    };
    fn_vect_.push_back({id, wrapped_function});
}

} // namespace holovibes