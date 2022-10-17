#include "API.hh"

namespace holovibes::api
{

void set_filter2d_n1(int value)
{
    api::detail::set_value<Filter2DN1>(value);
    api::set_auto_contrast_all();
}

void set_filter2d_n2(int value)
{
    api::detail::set_value<Filter2DN2>(value);
    api::set_auto_contrast_all();
}

} // namespace holovibes::api
