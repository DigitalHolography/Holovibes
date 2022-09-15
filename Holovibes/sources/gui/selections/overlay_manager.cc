#include "API.hh"
#include "overlay_manager.hh"
#include "zoom_overlay.hh"
#include "noise_overlay.hh"
#include "signal_overlay.hh"
#include "cross_overlay.hh"
#include "slice_cross_overlay.hh"
#include "composite_area_overlay.hh"
#include "rainbow_overlay.hh"
#include "reticle_overlay.hh"
#include "filter2d_reticle_overlay.hh"
#include "logger.hh"

namespace holovibes
{
namespace gui
{
OverlayManager::OverlayManager(BasicOpenGLWindow* parent)
    : current_overlay_(nullptr)
    , parent_(parent)
{
}

OverlayManager::~OverlayManager()
{
    units::RectFd empty_zone;
    api::set_signal_zone(empty_zone);
    api::set_noise_zone(empty_zone);
}

template <KindOfOverlay ko>
void OverlayManager::create_overlay()
{
    return;
}

template <>
void OverlayManager::create_overlay<Zoom>()
{
    if (!set_current(Zoom))
        create_overlay(std::make_shared<ZoomOverlay>(parent_));
}

template <>
void OverlayManager::create_overlay<Noise>()
{
    if (!set_current(Noise))
    {
        std::shared_ptr<Overlay> noise_overlay = std::make_shared<NoiseOverlay>(parent_);
        create_overlay(noise_overlay);
    }
}

template <>
void OverlayManager::create_overlay<Signal>()
{
    if (!set_current(Signal))
    {
        std::shared_ptr<Overlay> signal_overlay = std::make_shared<SignalOverlay>(parent_);
        create_overlay(signal_overlay);
    }
}

template <>
void OverlayManager::create_overlay<Cross>()
{
    if (!set_current(Cross))
        create_overlay(std::make_shared<CrossOverlay>(parent_));
    create_overlay<Zoom>();
}

template <>
void OverlayManager::create_overlay<SliceCross>()
{
    if (!set_current(SliceCross))
        create_overlay(std::make_shared<SliceCrossOverlay>(parent_));
}

template <>
void OverlayManager::create_overlay<KindOfOverlay::CompositeArea>()
{
    if (!set_current(KindOfOverlay::CompositeArea))
        create_overlay(std::make_shared<CompositeAreaOverlay>(parent_));
}

template <>
void OverlayManager::create_overlay<Rainbow>()
{
    if (!set_current(KindOfOverlay::Rainbow))
        create_overlay(std::make_shared<RainbowOverlay>(parent_));
}

template <>
void OverlayManager::create_overlay<Reticle>()
{
    if (!set_current(KindOfOverlay::Reticle))
        create_overlay(std::make_shared<ReticleOverlay>(parent_));
}

template <>
void OverlayManager::create_overlay<Filter2DReticle>()
{
    if (!set_current(KindOfOverlay::Filter2DReticle))
        create_overlay(std::make_shared<Filter2DReticleOverlay>(parent_));
}

void OverlayManager::create_overlay(std::shared_ptr<Overlay> new_overlay)
{
    clean();
    overlays_.push_back(new_overlay);
    set_current(new_overlay);
    current_overlay_->initProgram();
}

bool OverlayManager::set_current(KindOfOverlay ko)
{
    for (auto o : overlays_)
        if (o->getKind() == ko && o->isActive())
        {
            set_current(o);
            return true;
        }
    return false;
}

void OverlayManager::set_current(std::shared_ptr<Overlay> new_overlay)
{
    current_overlay_ = new_overlay;
    current_overlay_->onSetCurrent();
    api::get_cd().notify_observers();
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
        if (!current_overlay_->isActive())
            create_default();
        else if (current_overlay_->getKind() == Noise)
            create_overlay<Signal>();
        else if (current_overlay_->getKind() == Signal)
            create_overlay<Noise>();
    }
}

bool OverlayManager::disable_all(KindOfOverlay ko)
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

bool OverlayManager::enable_all(KindOfOverlay ko)
{
    bool found = false;
    for (auto o : overlays_)
        if (o->getKind() == ko)
        {
            o->enable();
            found = true;
        }
    return found;
}

void OverlayManager::draw()
{
    for (auto o : overlays_)
        if (o->isActive() && o->isDisplayed())
            o->draw();
}

void OverlayManager::clean()
{
    // Delete all disabled overlays
    overlays_.erase(std::remove_if(overlays_.begin(),
                                   overlays_.end(),
                                   [](std::shared_ptr<Overlay> overlay) { return !overlay->isActive(); }),
                    overlays_.end());
}

void OverlayManager::reset(bool def)
{
    for (auto o : overlays_)
        o->disable();
    if (def)
        create_default();
}

void OverlayManager::create_default()
{
    switch (parent_->getKindOfView())
    {
    case KindOfView::Filter2D:
        create_overlay<Filter2DReticle>();
    case KindOfView::Raw:
    case KindOfView::Hologram:
        create_overlay<Zoom>();
        break;
    case KindOfView::SliceXZ:
    case KindOfView::SliceYZ:
        create_overlay<SliceCross>();
        break;
    default:
        break;
    }
}

units::RectWindow OverlayManager::getZone() const
{
    CHECK(current_overlay_ != nullptr, "Overlay should never be null");
    return current_overlay_->getZone();
}

KindOfOverlay OverlayManager::getKind() const { return current_overlay_ ? current_overlay_->getKind() : Zoom; }

#ifdef _DEBUG
void OverlayManager::printVector()
{
    LOG_INFO(main, "Current overlay :");
    if (current_overlay_)
        current_overlay_->print();
    for (auto o : overlays_)
        o->print();
}
#endif
} // namespace gui
} // namespace holovibes
