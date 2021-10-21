#pragma once

#include <map>
#include <set>
#include <type_traits>
namespace holovibes
{

struct MicroCache
{
    void synchronize();

  protected:
    template <class T, class First, class... Args>
    void register_cache(First& elem, Args&&... args);

    template <class T, class First>
    void register_cache(First& elem);

    MicroCache();

    ~MicroCache();

    static inline std::set<MicroCache*> micro_caches_;

    std::map<std::string, void*> elem_to_ptr_;
    std::map<std::string, size_t> elem_to_size_;
    std::vector<std::pair<std::string, void*>> to_update;

    template <typename T>
    void need_update(const std::string& name, void* ptr);
};
} // namespace holovibes

#include "micro_cache.hxx"
