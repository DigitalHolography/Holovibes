#include "windowpostprocess_api.hh"

namespace holovibes::api
{

#pragma region Geometry Tr.

void set_rotation(WindowKind kind, float value)
{
    NOT_FILTER2D(kind, "rotation");

    auto window = get_window_xyz(kind);
    window.rotation = value;
    set_window_xyz(kind, window);

    pipe_refresh();
}

void set_horizontal_flip(WindowKind kind, bool value)
{
    NOT_FILTER2D(kind, "horizontal flip");
    auto window = get_window_xyz(kind);
    window.horizontal_flip = value;
    set_window_xyz(kind, window);
}

void set_horizontal_flip(bool value) { set_horizontal_flip(get_current_window_type(), value); }
void set_rotation(float value) { set_rotation(get_current_window_type(), value); }

#pragma endregion

#pragma region Accumulation

void set_accumulation_level(WindowKind kind, uint value)
{
    NOT_FILTER2D(kind, "accumulation");

    if (get_compute_mode() == Computation::Raw)
        return;

    auto window = get_window_xyz(kind);
    window.output_image_accumulation = value;
    set_window_xyz(kind, window);

    pipe_refresh();
}

void set_accumulation_level(uint value) { set_accumulation_level(get_current_window_type(), value); }

#pragma endregion

} // namespace holovibes::api