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
    fill_alpha_ = 0.1f;
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

        API.transform.set_p_index(slice_xz ? pIndex_.y() : pIndex_.x());
    }
}

void SliceCrossOverlay::release(ushort frameside) {}

void SliceCrossOverlay::setBuffer()
{
    bool slice_xz = parent_->getKindOfView() == KindOfView::SliceXZ;

    uint pmin = API.transform.get_p_index();
    uint pmax = pmin + API.transform.get_p_accu_level() + 1;

    units::PointFd topLeft = slice_xz ? units::PointFd(0, pmin) : units::PointFd(pmin, 0);
    units::PointFd bottomRight =
        slice_xz ? units::PointFd(parent_->getFd().width, pmax) : units::PointFd(pmax, parent_->getFd().height);
    zone_ = units::RectFd(topLeft, bottomRight);

    // Updating opengl buffer
    RectOverlay::setBuffer();
}
} // namespace holovibes::gui
