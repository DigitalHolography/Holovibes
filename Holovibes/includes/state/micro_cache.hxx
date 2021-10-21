#pragma once

#include "micro_cache.hh"

#define MONITORED_MEMBER(type, var)                                                                                    \
  private:                                                                                                             \
    struct var##_t                                                                                                     \
    {                                                                                                                  \
        type obj;                                                                                                      \
    };                                                                                                                 \
    var##_t var;                                                                                                       \
                                                                                                                       \
    void set_##var(type _val)                                                                                          \
    {                                                                                                                  \
        var.obj = _val;                                                                                                \
        need_update<std::remove_reference_t<decltype(*this)>>(#var, &var.obj);                                         \
    }                                                                                                                  \
                                                                                                                       \
  public:                                                                                                              \
    type get_##var() { return var.obj; }

namespace holovibes
{
template <class T, class First, class... Args>
void MicroCache::register_cache(First& elem, Args&&... args)
{
    const std::string name = std::string(typeid(T).name()) + "::" + typeid(First).name();
    elem_to_ptr_[name] = &elem;
    elem_to_size_[name] = sizeof(First);
    register_cache<T, Args...>(std::forward<Args>(args)...);
}

template <class T, class First>
void MicroCache::register_cache(First& elem)
{
    const std::string name = std::string(typeid(T).name()) + "::" + typeid(First).name();
    elem_to_ptr_[name] = &elem;
    elem_to_size_[name] = sizeof(First);
}

template <typename T>
void MicroCache::need_update(const std::string& name, void* ptr)
{
    for (MicroCache* cache : micro_caches_)
    {
        T* underlying_cache = reinterpret_cast<T*>(cache);
        if (underlying_cache == nullptr)
            return;

        underlying_cache->to_update.emplace_back(name, ptr);
    }
}

} // namespace holovibes
