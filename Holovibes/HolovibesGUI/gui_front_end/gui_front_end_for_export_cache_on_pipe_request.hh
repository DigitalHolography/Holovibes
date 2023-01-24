#pragma once

#include "all_caches.hh"

namespace holovibes::gui
{
class GuiFrontEndForExportCacheOnPipeRequest
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
    void after_method<Record>();
};
} // namespace holovibes::gui