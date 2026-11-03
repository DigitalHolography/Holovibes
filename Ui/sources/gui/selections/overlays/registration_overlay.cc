#include "API.hh"
#include "registration_overlay.hh"
#include "BasicOpenGLWindow.hh"

#include <algorithm>
#include <cmath>

namespace holovibes::gui
{
RegistrationOverlay::RegistrationOverlay(BasicOpenGLWindow* parent)
    : CircOverlay(Registration, parent, 80)
{
    LOG_FUNC();

    display_ = true;
    color_ = {1.f, 0.f, 0.f};
}

void RegistrationOverlay::setBuffer()
{
    const float scale = API.global_pp.get_registration_zone();

    scale_.x = scale;
    scale_.y = scale;

    if (API.transform.get_time_transformation() == TimeTransformation::LSCI)
    {
        // Match the reduced registration disk, which leaves room for the motion search.
        const auto& fd = parent_->getFd();
        const double resize_scale = std::min(1., 768. / std::max(fd.width, fd.height));
        const int width = std::max(1, static_cast<int>(std::lrint(fd.width * resize_scale)));
        const int height = std::max(1, static_cast<int>(std::lrint(fd.height * resize_scale)));
        const int side = std::min(width, height);
        if (side < 8)
            return;

        const int search_radius = std::max(1, static_cast<int>(std::lrint(.14 * side)));
        const int max_radius = side / 2 - search_radius - 1;
        const int radius = std::clamp(static_cast<int>(std::lrint(max_radius * scale)), 2, max_radius);
        scale_.x = 2.f * radius / width;
        scale_.y = 2.f * radius / height;
    }
}
} // namespace holovibes::gui
