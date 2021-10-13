#pragma once

#include "file_exception.hh"

namespace holovibes::io_files
{
inline FileException::FileException(const std::string& error_msg, bool display_errno)
    : std::exception()
    , error_msg_(error_msg)
{
    if (display_errno)
        error_msg_ = error_msg_ + ": " + std::strerror(errno);
}

inline const char* FileException::what() const noexcept
{
    return error_msg_.c_str();
}
} // namespace holovibes::io_files
