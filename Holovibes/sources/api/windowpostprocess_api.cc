#include "windowpostprocess_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Geometry Tr.

bool WindowPostProcessApi::get_horizontal_flip() { return get_horizontal_flip(api_->view.get_current_window_type()); }

void WindowPostProcessApi::set_horizontal_flip(WindowKind kind, bool value)
{
    NOT_FILTER2D(kind, "horizontal flip");
    auto window = get_window_xyz(kind);
    window.horizontal_flip = value;
    set_window_xyz(kind, window);
}

void WindowPostProcessApi::set_horizontal_flip(bool value)
{
    set_horizontal_flip(api_->view.get_current_window_type(), value);
}

float WindowPostProcessApi::get_rotation() { return get_rotation(api_->view.get_current_window_type()); }

void WindowPostProcessApi::set_rotation(WindowKind kind, float value)
{
    NOT_FILTER2D(kind, "rotation");

    auto window = get_window_xyz(kind);
    window.rotation = value;
    set_window_xyz(kind, window);

    api_->compute.pipe_refresh();
}

void WindowPostProcessApi::set_rotation(float value) { set_rotation(api_->view.get_current_window_type(), value); }

#pragma endregion

#pragma region Accumulation

uint WindowPostProcessApi::get_accumulation_level()
{
    return get_accumulation_level(api_->view.get_current_window_type());
}

void WindowPostProcessApi::set_accumulation_level(WindowKind kind, uint value)
{
    NOT_FILTER2D(kind, "accumulation");

    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    auto window = get_window_xyz(kind);
    window.output_image_accumulation = value;
    set_window_xyz(kind, window);

    api_->compute.pipe_refresh();
}

void WindowPostProcessApi::set_accumulation_level(uint value)
{
    set_accumulation_level(api_->view.get_current_window_type(), value);
}

#pragma endregion

#pragma region Enabled

bool WindowPostProcessApi::get_enabled() { return get_enabled(api_->view.get_current_window_type()); }

void WindowPostProcessApi::set_enabled(bool value) { set_enabled(api_->view.get_current_window_type(), value); }

#pragma endregion

} // namespace holovibes::api