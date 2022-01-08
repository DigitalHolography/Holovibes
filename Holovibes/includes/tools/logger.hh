#pragma once

#include <string>

#include <exception>
#include <fstream>
#include <sstream>
#include <functional>
#include <iostream>

#include <glog/logging.h>

#define LOG_TRACE DLOG(INFO) // Compat
#define LOG_DEBUG DLOG(INFO) // Compat
#define LOG_INFO LOG(INFO)
#define LOG_WARN LOG(WARNING)
#define LOG_ERROR LOG(ERROR)
#define LOG_FATAL LOG(FATAL)

// See glog doc for everything else
// CHECK(x) : assert + stacktrace
// PCHECK(x) : assert + stacktrace + check errno value
// CHECK_NOTNULL: ...
