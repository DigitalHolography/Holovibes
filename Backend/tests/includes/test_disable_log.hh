#pragma once
#include "holovibes.hh"

// This is usefull to disable logging while doing tests
namespace
{
struct DisableLogging
{
    DisableLogging() { holovibes::Logger::logger().get()->set_level(spdlog::level::off); }
};

static DisableLogging disableLoggingInstance; // Automatically runs at startup

} // namespace