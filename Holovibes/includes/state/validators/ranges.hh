#pragma once

#include "validators/validator.hh"

namespace holovibes::validators
{

template <typename T, int Min>
class Min : Validator<T>
{
    bool validate(T value) override const noexcept { return Min <= value; }
};

template <typename T, int Max>
class Max : Validator<T>
{
    bool validate(T value) override const noexcept { return value <= Max; }
};

template <typename T, int Min, int Max>
class Range : Validator<T>
{
    bool validate(T value) override const noexcept { return Min <= value <= Max; }
};

} // namespace holovibes::validators