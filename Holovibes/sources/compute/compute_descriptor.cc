#include "compute_descriptor.hh"
#include "user_interface_descriptor.hh"

#include "holovibes.hh"
#include "tools.hh"
#include "API.hh"

namespace holovibes
{
using LockGuard = std::lock_guard<std::mutex>;

ComputeDescriptor::ComputeDescriptor()
    : Observable()
{
}

ComputeDescriptor::~ComputeDescriptor() {}

void ComputeDescriptor::signalZone(units::RectFd& rect, AccessMode m)
{
    LockGuard g(mutex_);
    if (m == AccessMode::Get)
        rect = signal_zone;
    else if (m == AccessMode::Set)
        signal_zone = rect;
}

void ComputeDescriptor::noiseZone(units::RectFd& rect, AccessMode m)
{
    LockGuard g(mutex_);
    if (m == AccessMode::Get)
        rect = noise_zone;
    else if (m == AccessMode::Set)
        noise_zone = rect;
}

units::RectFd ComputeDescriptor::getCompositeZone() const
{
    LockGuard g(mutex_);
    return composite_zone;
}

void ComputeDescriptor::setCompositeZone(const units::RectFd& rect)
{
    LockGuard g(mutex_);
    composite_zone = rect;
}

units::RectFd ComputeDescriptor::getZoomedZone() const
{
    LockGuard g(mutex_);
    return zoomed_zone;
}

void ComputeDescriptor::setZoomedZone(const units::RectFd& rect)
{
    LockGuard g(mutex_);
    zoomed_zone = rect;
}

void ComputeDescriptor::setReticleZone(const units::RectFd& rect)
{
    LockGuard g(mutex_);
    reticle_zone = rect;
}

units::RectFd ComputeDescriptor::getReticleZone() const
{
    LockGuard g(mutex_);
    return reticle_zone;
}

float ComputeDescriptor::get_truncate_contrast_max(const int precision) const
{
    float value = GSH::instance().get_contrast_max();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

float ComputeDescriptor::get_truncate_contrast_min(const int precision) const
{
    float value = GSH::instance().get_contrast_min();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

void ComputeDescriptor::check_p_limits()
{
    int upper_bound = GSH::instance().get_time_transformation_size() - 1;

    if (GSH::instance().get_p_accu_level() > upper_bound)
        api::set_p_accu_level(upper_bound);

    upper_bound -= GSH::instance().get_p_accu_level();

    if (upper_bound >= 0 && GSH::instance().get_p_index() > static_cast<uint>(upper_bound))
        api::set_p_index(upper_bound);
}

void ComputeDescriptor::check_q_limits()
{
    int upper_bound = GSH::instance().get_time_transformation_size() - 1;

    if (GSH::instance().get_q_accu_level() > upper_bound)
        api::set_q_accu_level(upper_bound);

    upper_bound -= GSH::instance().get_q_accu_level();

    if (upper_bound >= 0 && GSH::instance().get_q_index() > static_cast<uint>(upper_bound))
        api::set_q_index(upper_bound);
}

void ComputeDescriptor::handle_update_exception()
{
    api::set_p_index(0);
    api::set_time_transformation_size(1);
    api::disable_convolution();
}

void ComputeDescriptor::handle_accumulation_exception() { GSH::instance().set_xy_img_accu_level(1); }

void ComputeDescriptor::change_angle()
{
    double rot = GSH::instance().get_rotation();
    double new_rot = (rot == 270.f) ? 0.f : rot + 90.f;

    GSH::instance().set_rotation(new_rot);
}

void ComputeDescriptor::change_flip() { GSH::instance().set_flip_enabled(!GSH::instance().get_flip_enabled()); }

void ComputeDescriptor::reset_windows_display()
{
    GSH::instance().set_lens_view_enabled(false);
    GSH::instance().set_filter2d_view_enabled(false);
    GSH::instance().set_raw_view_enabled(false);
    GSH::instance().set_reticle_display_enabled(false);
}

void ComputeDescriptor::reset_slice_view()
{
    GSH::instance().set_xz_contrast_max(false);
    GSH::instance().set_yz_contrast_max(false);

    GSH::instance().set_xz_log_scale_slice_enabled(false);
    GSH::instance().set_yz_log_scale_slice_enabled(false);

    GSH::instance().set_xz_img_accu_level(1);
    GSH::instance().set_yz_img_accu_level(1);
}

void ComputeDescriptor::set_divide_by_convo(bool enable)
{
    GSH::instance().set_divide_convolution_enabled(enable && GSH::instance().get_convolution_enabled());
}

} // namespace holovibes
