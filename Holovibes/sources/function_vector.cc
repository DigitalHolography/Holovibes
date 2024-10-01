#include "function_vector.hh"

namespace holovibes
{
FunctionVector::FunctionVector(ConditionType condition)
    : FnVector()
    , condition_(condition)
{
}

void FunctionVector::conditional_push_back(const FnType& function)
{
    push_back(
        [=]()
        {
            if (!condition_())
                return;

            function();
        });
}
} // namespace holovibes
