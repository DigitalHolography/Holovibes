#include "windowpostprocess_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Geometry Tr.

void WindowPostProcessApi::set_horizontal_flip(bool value, WindowKind kind) const
{
    NOT_FILTER2D(kind, "horizontal flip");
    auto window = get_window_xyz(kind);
    window.horizontal_flip = value;
    set_window_xyz(kind, window);
}

void WindowPostProcessApi::set_rotation(float value, WindowKind kind) const
{
    NOT_FILTER2D(kind, "rotation");

    auto window = get_window_xyz(kind);
    window.rotation = value;
    set_window_xyz(kind, window);
}

#pragma endregion

#pragma region Accumulation

void WindowPostProcessApi::set_accumulation_level(uint value, WindowKind kind) const
{
    NOT_FILTER2D(kind, "accumulation");

    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    auto window = get_window_xyz(kind);
    window.output_image_accumulation = value;
    set_window_xyz(kind, window);
}

#pragma endregion

} // namespace holovibes::api