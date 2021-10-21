#pragma once

#include "micro_cache.hh"

namespace holovibes
{

struct TestCache : public MicroCache
{
    MONITORED_MEMBER(int, a)
    MONITORED_MEMBER(float, b)
    MONITORED_MEMBER(std::string, c)

    friend class GSH;

    explicit TestCache() { register_cache<TestCache>(a, b, c); }
};
} // namespace holovibes
