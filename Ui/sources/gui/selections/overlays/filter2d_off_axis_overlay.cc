#include "filter2d_off_axis_overlay.hh"

#include <algorithm>

#include "API.hh"
#include "BasicOpenGLWindow.hh"
namespace holovibes::gui
{
namespace
{
inline int clamp_and_fix(int value, int min_value, int max_value)
{
    if (max_value < min_value)
        max_value = min_value;
    return std::clamp(value, min_value, max_value);
}
} // namespace

Filter2DOffAxisOverlay::Filter2DOffAxisOverlay(BasicOpenGLWindow* parent)
    : SquareOverlay(KindOfOverlay::Filter2DOffAxis, parent)
{
    color_ = {1.0f, 0.78f, 0.0f};
    alpha_ = 0.9f;
    apply_settings();
}

void Filter2DOffAxisOverlay::onSetCurrent()
{
    SquareOverlay::onSetCurrent();
    apply_settings();
}

void Filter2DOffAxisOverlay::apply_settings()
{
    const auto& fd = parent_->getFd();
    if (fd.width == 0 || fd.height == 0)
        return;

    int x_min = clamp_and_fix(API.filter2d.get_filter2d_off_axis_x_min(), 0, fd.width - 1);
    int x_max = clamp_and_fix(API.filter2d.get_filter2d_off_axis_x_max(), 0, fd.width - 1);
    int y_min = clamp_and_fix(API.filter2d.get_filter2d_off_axis_y_min(), 0, fd.height - 1);
    int y_max = clamp_and_fix(API.filter2d.get_filter2d_off_axis_y_max(), 0, fd.height - 1);

    if (x_max < x_min)
        std::swap(x_min, x_max);
    if (y_max < y_min)
        std::swap(y_min, y_max);

    // Convert inclusive max bounds to exclusive for the overlay representation.
    int x_max_exclusive = clamp_and_fix(x_max + 1, 0, fd.width);
    int y_max_exclusive = clamp_and_fix(y_max + 1, 0, fd.height);

    zone_.set_src(units::PointFd(x_min, y_min));
    zone_.set_dst(units::PointFd(x_max_exclusive, y_max_exclusive));

    make_square();
    ensure_square_within_bounds();
    setBuffer();
    display_ = true;
}

void Filter2DOffAxisOverlay::ensure_square_within_bounds()
{
    const auto& fd = parent_->getFd();
    if (fd.width == 0 || fd.height == 0)
        return;

    checkCorners();

    const int max_width = fd.width;
    const int max_height = fd.height;

    int x_min = clamp_and_fix(zone_.x(), 0, max_width - 1);
    int y_min = clamp_and_fix(zone_.y(), 0, max_height - 1);

    int width = std::max(1, std::min(zone_.unsigned_width(), zone_.unsigned_height()));
    int x_max_exclusive = x_min + width;
    int y_max_exclusive = y_min + width;

    if (x_max_exclusive > max_width)
    {
        x_max_exclusive = max_width;
        x_min = std::max(0, x_max_exclusive - width);
    }
    if (y_max_exclusive > max_height)
    {
        y_max_exclusive = max_height;
        y_min = std::max(0, y_max_exclusive - width);
    }

    zone_.set_src(units::PointFd(x_min, y_min));
    zone_.set_dst(units::PointFd(x_max_exclusive, y_max_exclusive));
    make_square();
    checkCorners();
}

void Filter2DOffAxisOverlay::release(ushort)
{
    make_square();
    ensure_square_within_bounds();

    const auto& fd = parent_->getFd();
    if (fd.width == 0 || fd.height == 0)
        return;

    const int x_min = clamp_and_fix(zone_.x(), 0, fd.width - 1);
    const int y_min = clamp_and_fix(zone_.y(), 0, fd.height - 1);
    const int x_max_exclusive = clamp_and_fix(zone_.right(), x_min + 1, fd.width);
    const int y_max_exclusive = clamp_and_fix(zone_.bottom(), y_min + 1, fd.height);

    int side = std::min(x_max_exclusive - x_min, y_max_exclusive - y_min);
    side = std::max(1, side);

    int x_max_inc = x_min + side - 1;
    int y_max_inc = y_min + side - 1;

    x_max_inc = clamp_and_fix(x_max_inc, x_min, fd.width - 1);
    y_max_inc = clamp_and_fix(y_max_inc, y_min, fd.height - 1);

    API.filter2d.set_filter2d_off_axis_x_min(x_min);
    API.filter2d.set_filter2d_off_axis_x_max(x_max_inc);
    API.filter2d.set_filter2d_off_axis_y_min(y_min);
    API.filter2d.set_filter2d_off_axis_y_max(y_max_inc);

    apply_settings();
}
} // namespace holovibes::gui
