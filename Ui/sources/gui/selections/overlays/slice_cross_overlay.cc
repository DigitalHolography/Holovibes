#include "slice_cross_overlay.hh"

#include "API.hh"
#include "BasicOpenGLWindow.hh"
#include "notifier.hh"
#include <math.h>

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
    LOG_ERROR("Dam in SliceCrossOverlay::keyPress ");

    if (e->key() == Qt::Key_Space)
    {
        LOG_ERROR("Dam in SliceCrossOverlay::keyPress keys_space");

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

        if (API.window_pp.get_horizontal_flip())
        {
            if (slice_xz)
            {
                pIndex_.set_y(parent_->getFd().height - pIndex_.y());
            }
            else
            {
                pIndex_.set_x(parent_->getFd().width - pIndex_.x());
            }
        }

        uint rot = API.window_pp.get_rotation();
        if (rot != 0)
        {
            double width = parent_->getFd().width;
            double height = parent_->getFd().height;
            double xc = width / 2.0;
            double yc = height / 2.0;

            double dx = pIndex_.x() - xc;
            double dy = pIndex_.y() - yc;
            double x_rot = pIndex_.x(), y_rot = pIndex_.y();

            switch (rot)
            {
            case 90:
                x_rot = xc - dy;
                y_rot = yc + dx;
                break;
            case 180:
                x_rot = xc - dx;
                y_rot = yc - dy;
                break;
            case 270:
                x_rot = xc + dy;
                y_rot = yc - dx;
                break;
            default:
                break;
            }
            pIndex_.set_x(x_rot);
            pIndex_.set_y(y_rot);
        }

        double p_index = slice_xz ? pIndex_.y() : pIndex_.x();
        API.transform.set_p_index(p_index);

        NotifierManager::notify("notify", true);
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
