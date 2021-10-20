#pragma once

#include <boost/pfr.hpp>

struct MicroCache
{
    // template <typename... Args>
    // MicroCached(const Args &... args)
    // {
    //     std::tuple<Args...> vars = boost::pfr::structure_tie(*this);
    //     for (auto x : args...)
    //     {

    //     }
    // }

    virtual void synchronize();
};
