#pragma once

#include <mutex>

namespace holovibes
{

class GSH
{

  private:
    GSH() {}

    static GSH* instance_;
    static std::mutex mutex_;

  public:
    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

    GSH& instance();
};
} // namespace holovibes
