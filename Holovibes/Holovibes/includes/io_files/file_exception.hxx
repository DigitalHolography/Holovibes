#pragma once

#include "file_exception.hh"

namespace holovibes::io_files
{
inline FileException::FileException(const std::string& error_msg,
                                    bool display_errno)
    : std::exception()
    , error_msg_(error_msg)
    , display_errno_(display_errno)
{
}

inline const char* FileException::what() const noexcept
{
    if (display_errno_)
        return (error_msg_ + ": " + std::strerror(errno)).c_str();

    return error_msg_.c_str();
}
} // namespace holovibes::io_files
