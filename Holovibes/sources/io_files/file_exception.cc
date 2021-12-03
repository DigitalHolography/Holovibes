#include "file_exception.hh"

namespace holovibes::io_files
{
FileException::FileException(const std::string& error_msg, bool display_errno)
    : std::exception()
    , error_msg_(error_msg)
{
    if (display_errno)
        error_msg_ = error_msg_ + ": " + std::strerror(errno);
}
} // namespace holovibes::io_files
