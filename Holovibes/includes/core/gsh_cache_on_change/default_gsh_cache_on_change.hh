#pragma once

#include "logger.hh"

#ifndef DISABLE_LOG_GSH_ON_CHANGE
#define LOG_ON_CHANGE_GSH(type) LOG_TRACE(main, "On Change : {}", typeid(type).name())
#else
#define LOG_ON_CHANGE_GSH(type)
#endif