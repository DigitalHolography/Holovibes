#pragma once

#include "global_state_holder.hh"
#include "micro_cache.hh"

namespace holovibes
{
struct ComputeCache : MicroCache
{
    ComputeCache(int a, float b, std::string c)
        : a_(a)
        , b_(b)
        , c_(c)
    {
        // GSH::cache_map<int>[typeid(a).name()][&a].push_back(&a_);
        // GSH::cache_map<float>[typeid(b).name()][&b].push_back(&b_);
        // GSH::cache_map<std::string>[typeid(c).name()][&c].push_back(&c_);

        GSH::cache_map[&a].push_back(&a_);
        GSH::cache_map[&b].push_back(&b_);
        GSH::cache_map[&c].push_back(&c_);
    }

    int a_;
    float b_;
    std::string c_;
}
