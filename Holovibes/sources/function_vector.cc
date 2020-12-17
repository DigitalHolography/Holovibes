/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
    push_back([=]() {
        if (!condition_())
            return;

        function();
    });
}
} // namespace holovibes