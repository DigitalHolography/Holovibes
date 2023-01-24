#pragma once

#include "logger.hh"
#include "compute_cache.hh"

namespace holovibes
{
class ComputeGSHOnChange
{
  public:
    template <typename T>
    void operator()(typename T::ValueType&)
    {
    }

    template <typename T>
    bool change_accepted(typename T::ConstRefType)
    {
        return true;
    }

  public:
    template <>
    bool change_accepted<TimeTransformationSize>(uint new_value)
    {
        return new_value != 0;
    }

  public:
    template <>
    void operator()<ComputeMode>(ComputeModeEnum& new_value);
    template <>
    void operator()<ImageType>(ImageTypeEnum& new_value);
    template <>
    void operator()<BatchSize>(int& new_value);
    template <>
    void operator()<TimeStride>(int& new_value);
    template <>
    void operator()<TimeTransformationCutsEnable>(bool& new_value);
    template <>
    void operator()<Filter2D>(Filter2DStruct& new_value);
    template <>
    void operator()<SpaceTransformation>(SpaceTransformationEnum& new_value);
};
} // namespace holovibes
