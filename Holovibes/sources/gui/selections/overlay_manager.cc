#include "API.hh"
#include "overlay_manager.hh"
#include "zoom_overlay.hh"
#include "stabilization_overlay.hh"
#include "noise_overlay.hh"
#include "signal_overlay.hh"
#include "cross_overlay.hh"
#include "slice_cross_overlay.hh"
#include "composite_area_overlay.hh"
#include "rainbow_overlay.hh"
#include "reticle_overlay.hh"
#include "logger.hh"

#include <QDateTime>

namespace holovibes::gui
{
OverlayManager::OverlayManager(BasicOpenGLWindow* parent)
    : current_overlay_(nullptr)
    , parent_(parent)
    , timer_(parent)
{
    parent_->connect(&timer_, &QTimer::timeout, parent_, [=] { OverlayManager::hide_overlay(); });
    timer_.start(100);
}

OverlayManager::~OverlayManager()
{
    units::RectFd empty_zone;
    api::set_signal_zone(empty_zone);
    api::set_noise_zone(empty_zone);
}

void OverlayManager::hide_overlay()
{
    QDateTime current_time = QDateTime::currentDateTime();
    for (auto o : overlays_)
        if (o->getTimeBeforeHide() < current_time)
            o->disable();
}

bool OverlayManager::disable(KindOfOverlay ko)
{
    bool found = false;
    for (auto o : overlays_)
        if (o->getKind() == ko)
        {
            o->disable();
            found = true;
        }

    return found;
}

void OverlayManager::clean()
{
    // Delete all disabled overlays
    overlays_.erase(std::remove_if(overlays_.begin(),
                                   overlays_.end(),
                                   [](std::shared_ptr<Overlay> overlay) { return !overlay->isActive(); }),
                    overlays_.end());
}

void OverlayManager::create_default()
{
    switch (parent_->getKindOfView())
    {
    case KindOfView::Raw:
    case KindOfView::Hologram:
        enable<Zoom>();
        break;
    case KindOfView::SliceXZ:
    case KindOfView::SliceYZ:
        enable<SliceCross>();
        break;
    default:
        break;
    }
}

void OverlayManager::press(QMouseEvent* e)
{
    if (current_overlay_)
        current_overlay_->press(e);
}

void OverlayManager::keyPress(QKeyEvent* e)
{
    // Reserving space for moving the cross
    if (e->key() == Qt::Key_Space)
    {
        for (auto o : overlays_)
            if ((o->getKind() == Cross || o->getKind() == SliceCross) && o->isActive())
                o->keyPress(e);
    }
    else if (current_overlay_)
        current_overlay_->keyPress(e);
}

void OverlayManager::move(QMouseEvent* e)
{
    for (auto o : overlays_)
        if ((o->getKind() == Cross || o->getKind() == SliceCross) && o->isActive())
            o->move(e);
    if (current_overlay_)
        current_overlay_->move(e);
}

void OverlayManager::release(ushort frame)
{
    if (current_overlay_)
    {
        current_overlay_->release(frame);

        if (current_overlay_->getKind() == Noise)
            enable<Signal>();
        else if (current_overlay_->getKind() == Signal)
            enable<Noise>();
    }
}

void OverlayManager::draw()
{
    for (auto o : overlays_)
        if (o->isActive() && o->isDisplayed())
            o->draw();
}

std::shared_ptr<Overlay> OverlayManager::create_overlay(KindOfOverlay ko)
{
    switch (ko)
    {
    case Zoom:
        return std::make_shared<ZoomOverlay>(parent_);
    case Stabilization:
        return std::make_shared<StabilizationOverlay>(parent_);
    case Noise:
        return std::make_shared<NoiseOverlay>(parent_);
    case Signal:
        return std::make_shared<SignalOverlay>(parent_);
    case Cross:
        return std::make_shared<CrossOverlay>(parent_);
    case SliceCross:
        return std::make_shared<SliceCrossOverlay>(parent_);
    case CompositeArea:
        return std::make_shared<CompositeAreaOverlay>(parent_);
    case Rainbow:
        return std::make_shared<RainbowOverlay>(parent_);
    case Reticle:
        return std::make_shared<ReticleOverlay>(parent_);
    default:
        return nullptr;
    }
}

KindOfOverlay OverlayManager::getKind() const { return current_overlay_ ? current_overlay_->getKind() : Zoom; }

#ifdef _DEBUG
void OverlayManager::printVector()
{
    LOG_INFO("Current overlay :");
    if (current_overlay_)
        current_overlay_->print();
    for (auto o : overlays_)
        o->print();
}
#endif
} // namespace holovibes::gui
