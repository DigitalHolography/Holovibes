#pragma once

#include <string>
#include "validators/validator.hh"

namespace holovibes::validators
{

template <typename T>
class NotEmpty : Validator<T>
{
    bool validate(T value) override const noexcept { return std::size(value) > 0; }
};

} // namespace holovibes::validators