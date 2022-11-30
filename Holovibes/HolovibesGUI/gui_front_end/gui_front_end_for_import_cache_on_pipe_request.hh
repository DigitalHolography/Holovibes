#pragma once

#include "all_caches.hh"

namespace holovibes::gui
{
class GuiFrontEndForImportCacheOnPipeRequest
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
    void before_method<ImportType>();
    template <>
    void after_method<ImportType>();
};
} // namespace holovibes::gui