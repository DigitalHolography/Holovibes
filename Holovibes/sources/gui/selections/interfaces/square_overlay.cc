#include "square_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
SquareOverlay::SquareOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent)
    : FilledRectOverlay(overlay, parent)
{
}

void SquareOverlay::make_square()
{
    // Set the bottomRight corner to have a square selection.
    const int min = std::min(std::abs(zone_.width()), std::abs(zone_.height()));
    zone_.setDst(units::PointFd(zone_.src().x() + ((zone_.src().x() < zone_.dst().x()) ? min : -min),
                                zone_.src().y() + ((zone_.src().y() < zone_.dst().y()) ? min : -min)));
}

void SquareOverlay::checkCorners()
{
    auto parent_fd = parent_->getFd();
    ushort frameSide = std::min(parent_fd.width, parent_fd.height);

    // Resizing the square selection to the window
    if (zone_.dst().x() < 0)
        zone_.dstRef().x() = 0;
    else if (zone_.dst().x() > frameSide)
        zone_.dstRef().x() = frameSide;

    if (zone_.dst().y() < 0)
        zone_.dstRef().y() = 0;
    else if (zone_.dst().y() > frameSide)
        zone_.dstRef().y() = frameSide;

    // Making it a square again
    make_square();
}

void SquareOverlay::move(QMouseEvent* e)
{
    if (e->buttons() == Qt::LeftButton)
    {
        auto pos = getMousePos(e->pos());
        zone_.setDst(pos);
        make_square();
        setBuffer();
        display_ = true;
    }
}
} // namespace holovibes::gui
