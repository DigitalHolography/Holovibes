#pragma once

#include <deque>
#include <vector>
#include <mutex>
#include <functional>

using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;

using LockGuard = std::lock_guard<std::mutex>;

using MutexGuard = std::lock_guard<std::mutex>;

using ConditionType = std::function<bool()>;

/*! \brief A single procedure. */
using FnType = std::function<void()>;

/*! \brief A procedure vector. */
using FnVector = std::vector<FnType>;

/*! \brief A procedure deque. */
using FnDeque = std::deque<FnType>;

using Color = std::array<float, 3>;

class ICompute;
using SharedPipe = std::shared_ptr<ICompute>;
