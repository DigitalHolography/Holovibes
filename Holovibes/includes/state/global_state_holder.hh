#pragma once

#include <mutex>

#include "fast_updates_holder.hh"
#include "compute_cache.hh"
#include <map>

namespace holovibes
{
struct ComputeCache;
class GSH
{
  public:
    enum class SyncTrigger
    {
        NOW,
        END_OF_PIPE,
    }

    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

    GSH& instance();

    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

    // template <class T>
    // static inline std::unordered_map<std::string, std::unordered_map<T*, std::vector<T*>>> cache_map;

    std::map<void*, size_t> elem_to_size_map;
    std::map<void*, SyncTrigger> elem_to_trigger_map;
    std::map<void*, std::vector<void*>> cache_map;

    void synchronize();

  private:
    GSH() {}

    int a_;
    float b_;
    std::string c_;

    int get_A() const { return a_; }

    void set_A(int a)
    {
        a_ = a;
        to_update.push_back(&a_);
    }

    float get_B() const { return b_; }

    void set_B(float b)
    {
        b_ = b;
        to_update.push_back(&b_);
    }

    std::string get_C() const { return c_; }

    void set_C(std::string c)
    {
        c_ = c;
        to_update.push_back(&c_);
    }

    ComputeCache compute_cache;

    std::mutex mutex_;

    static inline std::vector<void*> to_update;
};

} // namespace holovibes
