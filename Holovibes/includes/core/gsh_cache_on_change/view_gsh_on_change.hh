#pragma once

#include "logger.hh"
#include "view_cache.hh"

namespace holovibes
{
class UserInterface;
class ViewGSHOnChange
{
  public:
    template <typename T>
    void operator()(typename T::ValueType&)
    {
    }

  public:
    template <typename T>
    bool change_accepted(typename T::ConstRefType)
    {
        return true;
    }

  public:
    template <>
    bool change_accepted<Reticle>(const ReticleStruct& new_value)
    {
        return new_value.scale >= 0. && new_value.scale <= 1.;
    }

    template <>
    bool change_accepted<RawViewEnabled>(bool new_value);
    template <>
    bool change_accepted<CutsViewEnabled>(bool new_value);
    template <>
    bool change_accepted<ChartDisplayEnabled>(bool new_value);
    template <>
    bool change_accepted<Filter2DViewEnabled>(bool new_value);
    template <>
    bool change_accepted<LensViewEnabled>(bool new_value);
    template <>
    bool change_accepted<ViewAccuP>(const ViewAccuPQ& new_value);
    template <>
    bool change_accepted<ViewAccuX>(const ViewAccuXY& new_value);
    template <>
    bool change_accepted<ViewAccuY>(const ViewAccuXY& new_value);

  public:
    template <>
    void operator()<ViewAccuX>(ViewAccuXY& new_value);
    template <>
    void operator()<ViewAccuY>(ViewAccuXY& new_value);
    template <>
    void operator()<ViewAccuP>(ViewAccuPQ& new_value);
    template <>
    void operator()<ViewAccuQ>(ViewAccuPQ& new_value);
    template <>
    void operator()<RawViewEnabled>(bool& new_value);
    template <>
    void operator()<CutsViewEnabled>(bool& new_value);
    template <>
    void operator()<ChartDisplayEnabled>(bool& new_value);
    template <>
    void operator()<LensViewEnabled>(bool& new_value);
    template <>
    void operator()<Filter2DViewEnabled>(bool& new_value);
};
} // namespace holovibes
