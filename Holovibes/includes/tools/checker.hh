#pragma once

#define LOGURU_WITH_STREAMS 1
#include "loguru.hpp"

#define CHECK(cond) DCHECK_S(cond)