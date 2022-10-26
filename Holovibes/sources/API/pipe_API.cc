#include "API.hh"

namespace holovibes::api
{

void create_pipe()
{
    try
    {
        Holovibes::instance().start_compute();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR(main, "cannot create Pipe: {}", e.what());
    }
}

} // namespace holovibes::api
