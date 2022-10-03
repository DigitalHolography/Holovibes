#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class fftShiftEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    fftShiftEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    fftShiftEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "fft_shift_enabled"; }
    const char* get_key() const override { return fftShiftEnabled::static_key(); }
};

} // namespace holovibes
