#include "delete_twin_overlay.hh"

#include "API.hh"
#include "BasicOpenGLWindow.hh"
#include "logger.hh"

namespace holovibes::gui
{
namespace
{
bool is_degenerate(const units::RectFd& rect) { return rect.x() == rect.right() || rect.y() == rect.bottom(); }
} // namespace

DeleteTwinMaskOverlay::DeleteTwinMaskOverlay(BasicOpenGLWindow* parent)
    : RectOverlay(KindOfOverlay::DeleteTwinMask, parent)
{
    color_ = {0.2f, 0.8f, 0.9f};
    alpha_ = 1.0f;
}

void DeleteTwinMaskOverlay::onSetCurrent()
{
    Overlay::onSetCurrent();

    const auto& fd = parent_->getFd();
    auto rect = API.transform.get_delete_twin_image_rectangle();

    if (is_degenerate(rect))
    {
        const int half_width = static_cast<int>(fd.width / 4);
        const int half_height = static_cast<int>(fd.height / 4);
        const int center_x = static_cast<int>(fd.width / 2);
        const int center_y = static_cast<int>(fd.height / 2);
        rect = units::RectFd(center_x - half_width,
                             center_y - half_height,
                             center_x + half_width,
                             center_y + half_height);
    }

    zone_ = rect;
    RectOverlay::checkCorners();
    setBuffer();
    display_ = true;
}

void DeleteTwinMaskOverlay::release(ushort)
{
    RectOverlay::checkCorners();
    setBuffer();

    if (is_degenerate(zone_))
        return;

    auto status = API.transform.set_delete_twin_image_rectangle(zone_);
    if (status != ApiCode::OK && status != ApiCode::NO_CHANGE)
        LOG_WARN("Unable to update delete twin mask rectangle (status: {})", static_cast<int>(status));

    display_ = true;
}
} // namespace holovibes::gui
