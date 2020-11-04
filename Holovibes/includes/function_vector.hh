#pragma once

#include "pipeline_utils.hh"

namespace holovibes
{
    using ConditionType = std::function<bool()>;

    class FunctionVector : public FnVector
    {
    public:
        FunctionVector(ConditionType condition);

        FunctionVector() = default;

        ~FunctionVector() = default;

        /*!
        ** \brief Push back the function in the vector.
        ** Execute it only if the condition is verified.
        */
        void conditional_push_back(const FnType& function);

    private:
        ConditionType condition_;
    };
}