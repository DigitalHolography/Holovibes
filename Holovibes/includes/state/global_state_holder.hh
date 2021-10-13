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

    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

  private:
    GSH() {}

    std::mutex mutex_;
};

} // namespace holovibes
