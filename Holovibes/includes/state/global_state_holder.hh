#pragma once

#include <mutex>

#include "fast_updates_holder.hh"

namespace holovibes
{

class GSH
{
  public:
    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

    GSH& instance();

    template <class T>
    static FastUpdatesHolder<T> fast_updates_map;

  private:
    GSH() {}

    static GSH* instance_;
    static std::mutex mutex_;
};
} // namespace holovibes
