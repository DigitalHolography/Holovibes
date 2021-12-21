#include "API.hh"
#include "units\conversion_data.hh"
#include "units\unit.hh"
#include "BasicOpenGLWindow.hh"
#include "global_state_holder.hh"

namespace holovibes
{
using gui::BasicOpenGLWindow;
namespace units
{
ConversionData::ConversionData(const BasicOpenGLWindow& window)
    : window_(&window)
{
}

ConversionData::ConversionData(const BasicOpenGLWindow* window)
    : window_(window)
{
}

float ConversionData::window_size_to_opengl(int val, Axis axis) const
{
    CHECK(window_ != nullptr) << "gui::BasicOpenGLWindow *window_ cannot be null";
    auto res = (static_cast<float>(val) * 2.f / static_cast<float>(get_window_size(axis))) - 1.f;
    return axis == Axis::VERTICAL ? -res : res;
}

float ConversionData::fd_to_opengl(int val, Axis axis) const
{
    CHECK(window_ != nullptr) << "gui::BasicOpenGLWindow *window_ cannot be null";
    auto res = (static_cast<float>(val) * 2.f / static_cast<float>(get_fd_size(axis))) - 1.f;
    return axis == Axis::VERTICAL ? -res : res;
}

int ConversionData::opengl_to_window_size(float val, Axis axis) const
{
    CHECK(window_ != nullptr) << "gui::BasicOpenGLWindow *window_ cannot be null";
    if (axis == Axis::VERTICAL)
        val *= -1;
    int res = ((val + 1.f) / 2.f) * get_window_size(axis);
    return res;
}

int ConversionData::opengl_to_fd(float val, Axis axis) const
{
    CHECK(window_ != nullptr) << "gui::BasicOpenGLWindow *window_ cannot be null";
    if (axis == Axis::VERTICAL)
        val *= -1;
    return ((val + 1.f) / 2.f) * get_fd_size(axis);
}

double ConversionData::fd_to_real(int val, Axis axis) const
{
    CHECK(window_ != nullptr) << "gui::BasicOpenGLWindow *window_ cannot be null";
    auto fd = window_->getFd();
    float pix_size;
    if (window_->getKindOfView() == gui::KindOfView::Hologram)
        pix_size = (GSH::instance().get_lambda() * GSH::instance().get_z_distance()) /
                   (fd.width * GSH::instance().get_pixel_size() * 1e-6);
    else if (window_->getKindOfView() == gui::KindOfView::SliceXZ && axis == Axis::HORIZONTAL)
    {
        pix_size = (GSH::instance().get_lambda() * GSH::instance().get_z_distance()) /
                   (fd.width * GSH::instance().get_pixel_size() * 1e-6);
    }
    else if (window_->getKindOfView() == gui::KindOfView::SliceYZ && axis == Axis::VERTICAL)
    {
        pix_size = (GSH::instance().get_lambda() * GSH::instance().get_z_distance()) /
                   (fd.height * GSH::instance().get_pixel_size() * 1e-6);
    }
    else
    {
        pix_size = std::pow(GSH::instance().get_lambda(), 2) / 50E-9; // 50nm is an arbitrary value
    }

    return val * pix_size;
}

void ConversionData::transform_from_fd(float& x, float& y) const
{
    glm::vec3 input{x, y, 1.0f};
    const auto& matrix = window_->getTransformMatrix();
    auto output = matrix * input;
    x = output[0];
    y = output[1];
}

void ConversionData::transform_to_fd(float& x, float& y) const
{
    glm::vec3 input{x, y, 1};
    auto matrix = window_->getTransformInverseMatrix();
    auto output = matrix * input;
    x = output[0];
    y = output[1];
}
int ConversionData::get_window_size(Axis axis) const
{
    switch (axis)
    {
    case Axis::HORIZONTAL:
        return window_->width();
    case Axis::VERTICAL:
        return window_->height();
    default:
        throw std::exception("Unreachable code");
    }
}

int ConversionData::get_fd_size(Axis axis) const
{
    switch (axis)
    {
    case Axis::HORIZONTAL:
        return window_->getFd().width;
    case Axis::VERTICAL:
        return window_->getFd().height;
    default:
        throw std::exception("Unreachable code");
    }
}
} // namespace units
} // namespace holovibes
