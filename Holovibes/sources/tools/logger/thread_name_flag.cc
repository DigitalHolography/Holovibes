#include "spdlog/pattern_formatter.h"
#include "thread_name_flag.hh"
#include "logger.hh"

namespace holovibes
{
std::string ThreadNameFlag::get_thread_name(size_t thread_id)
{
    std::pair<std::string, bool> search = Logger::get_thread_name(thread_id);
    if (search.second)
    {
        return search.first;
    }
    else
    {
        return std::to_string(thread_id);
    }
}

void ThreadNameFlag::format(const spdlog::details::log_msg& log_msg, const std::tm&, spdlog::memory_buf_t& dest)
{
    std::string thread_name = get_thread_name(log_msg.thread_id);
    dest.append(thread_name.data(), thread_name.data() + thread_name.size());
}

std::unique_ptr<spdlog::custom_flag_formatter> ThreadNameFlag::clone() const
{
    return spdlog::details::make_unique<ThreadNameFlag>();
}
} // namespace holovibes
