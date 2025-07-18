#include "square_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "API.hh"

namespace holovibes::gui
{
SquareOverlay::SquareOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent)
    : FilledRectOverlay(overlay, parent)
{
}

void SquareOverlay::make_square()
{
    // Set the bottom right corner to have a square selection.
    const auto& fd = API.input.get_input_fd();
    const float frameW = static_cast<float>(fd.width);
    const float frameH = static_cast<float>(fd.height);

    const float w = std::abs(zone_.width());
    const float h = std::abs(zone_.height());

    float newW, newH;
    if (w * frameH < h * frameW)
    {
        newW = w;
        newH = w * frameH / frameW;
    }
    else
    {
        newW = h * frameW / frameH;
        newH = h;
    }

    if (zone_.dst().x() < zone_.src().x())
        newW = -newW;
    if (zone_.dst().y() < zone_.src().y())
        newH = -newH;

    zone_.set_dst(units::PointFd(zone_.src().x() + newW, zone_.src().y() + newH));
}

void SquareOverlay::checkCorners()
{
    auto parent_fd = parent_->getFd();
    ushort frameSide = std::min(parent_fd.width, parent_fd.height);

    // Resizing the square selection to the window
    if (zone_.dst().x() < 0)
        zone_.dst_ref().x() = 0;
    else if (zone_.dst().x() > frameSide)
        zone_.dst_ref().x() = frameSide;

    if (zone_.dst().y() < 0)
        zone_.dst_ref().y() = 0;
    else if (zone_.dst().y() > frameSide)
        zone_.dst_ref().y() = frameSide;

    // Making it a square again
    make_square();
}

void SquareOverlay::move(QMouseEvent* e)
{
    if (e->buttons() == Qt::LeftButton)
    {
        auto pos = getMousePos(e->pos());
        zone_.set_dst(pos);
        make_square();
        setBuffer();
        display_ = true;
    }
}
} // namespace holovibes::gui
