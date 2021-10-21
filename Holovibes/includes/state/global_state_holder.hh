#pragma once

#include <mutex>

#include "fast_updates_holder.hh"
#include "struct_test.hh"

namespace holovibes
{

class GSH
{
  public:
    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

    static GSH& instance();

    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

    // std::map<std::string, void*> elem_to_ptr_;
    // std::vector<std::string> to_update_;

    // void query(TestCacheQuery test_cache_query) { test_cache_.set_a(test_cache_query.a); }

  private:
    GSH() {}

    mutable std::mutex mutex_;
};

} // namespace holovibes
