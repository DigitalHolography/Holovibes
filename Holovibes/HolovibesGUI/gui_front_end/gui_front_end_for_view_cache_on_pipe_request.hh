#pragma once

#include "view_cache.hh"

namespace holovibes::gui
{
class GuiFrontEndForViewCacheOnPipeRequest
{
  public:
    template <typename T>
    static void before_method()
    {
    }
    template <typename T>
    static void after_method()
    {
    }

  public:
    template <>
    void before_method<ChartDisplayEnabled>();
    template <>
    void after_method<ChartDisplayEnabled>();

    template <>
    void before_method<CutsViewEnable>();
    template <>
    void after_method<CutsViewEnable>();

    template <>
    void before_method<Reticle>();
    template <>
    void after_method<Reticle>();

    template <>
    void before_method<LensViewEnabled>();
    template <>
    void after_method<LensViewEnabled>();

    template <>
    void before_method<RawViewEnabled>();
    template <>
    void after_method<RawViewEnabled>();

    template <>
    void before_method<Filter2DViewEnabled>();
    template <>
    void after_method<Filter2DViewEnabled>();
};
} // namespace holovibes::gui