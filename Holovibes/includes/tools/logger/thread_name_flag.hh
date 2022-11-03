#include "spdlog/pattern_formatter.h"

namespace holovibes
{
class ThreadNameFlag : public spdlog::custom_flag_formatter
{
  private:
    std::string get_thread_name(size_t thread_id);

  public:
    void format(const spdlog::details::log_msg&, const std::tm&, spdlog::memory_buf_t& dest) override;
    std::unique_ptr<spdlog::custom_flag_formatter> clone() const override;
};
} // namespace holovibes