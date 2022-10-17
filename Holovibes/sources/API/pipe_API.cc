#include "API.hh"

namespace holovibes::api
{

void pipe_refresh()
{
    if (get_compute_mode() == Computation::Raw || UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    try
    {
        api::get_compute_pipe().request_refresh();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR(compute_worker, "{}", e.what());
    }
}

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
