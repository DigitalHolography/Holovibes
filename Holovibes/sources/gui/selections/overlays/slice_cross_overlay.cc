#include "API.hh"
#include "slice_cross_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "API.hh"

namespace holovibes::gui
{
SliceCrossOverlay::SliceCrossOverlay(BasicOpenGLWindow* parent)
    : FilledRectOverlay(KindOfOverlay::SliceCross, parent)
    , locked_(true)
    , pIndex_(0, 0)
{
    display_ = true;
    fill_alpha_ = 0.15f;
}

void SliceCrossOverlay::keyPress(QKeyEvent* e)
{
    if (e->key() == Qt::Key_Space)
    {
        locked_ = !locked_;
        parent_->setCursor(locked_ ? Qt::ArrowCursor : Qt::CrossCursor);
    }
}

void SliceCrossOverlay::move(QMouseEvent* e)
{
    if (!locked_)
    {
        bool slice_xz = parent_->getKindOfView() == KindOfView::SliceXZ;

        pIndex_ = getMousePos(e->pos());

        api::set_p_index(slice_xz ? pIndex_.y() : pIndex_.x());
    }
}

void SliceCrossOverlay::release(ushort frameside) {}

void SliceCrossOverlay::setBuffer()
{
    bool slice_xz = parent_->getKindOfView() == KindOfView::SliceXZ;

    ViewPQ p = api::get_p();

    uint pmin = p.start;
    uint pmax = pmin + p.width + 1;

    units::ConversionData convert(parent_);

    units::PointFd topLeft = slice_xz ? units::PointFd(convert, 0, pmin) : units::PointFd(convert, pmin, 0);
    units::PointFd bottomRight = slice_xz ? units::PointFd(convert, parent_->getFd().width, pmax)
                                          : units::PointFd(convert, pmax, parent_->getFd().height);
    zone_ = units::RectFd(topLeft, bottomRight);

    // Updating opengl buffer
    RectOverlay::setBuffer();
}
} // namespace holovibes::gui
