#include "API.hh"
#include "units\conversion_data.hh"
#include "units\unit.hh"
#include "BasicOpenGLWindow.hh"
#include "logger.hh"

namespace holovibes
{
using gui::BasicOpenGLWindow;
} // namespace holovibes

namespace holovibes::units
{
ConversionData::ConversionData(const BasicOpenGLWindow& window)
    : window_(&window)
{
}

ConversionData::ConversionData(const BasicOpenGLWindow* window)
    : window_(window)
{
}

float ConversionData::fd_to_opengl(int val, Axis axis) const
{
    CHECK(window_ != nullptr, "gui::BasicOpenGLWindow *window_ cannot be null");
    auto res = (static_cast<float>(val) * 2.f / static_cast<float>(get_fd_size(axis))) - 1.f;
    return axis == Axis::VERTICAL ? -res : res;
}

int ConversionData::opengl_to_fd(float val, Axis axis) const
{
    CHECK(window_ != nullptr, "gui::BasicOpenGLWindow *window_ cannot be null");
    if (axis == Axis::VERTICAL)
        val *= -1;
    return ((val + 1.f) / 2.f) * get_fd_size(axis);
}

void ConversionData::transform_from_fd(float& x, float& y) const
{
    glm::vec3 input{x, y, 1.0f};
    const auto& matrix = window_->getTransformMatrix();
    auto output = matrix * input;
    x = output[0];
    y = output[1];
    LOG_ERROR("From FD");
    LOG_ERROR("From FD 1");
}

void ConversionData::transform_to_fd(float& x, float& y) const
{
    glm::vec3 input{x, y, 1};
    auto matrix = window_->getTransformInverseMatrix();
    auto output = matrix * input;
    x = output[0];
    y = output[1];
    LOG_ERROR("To FD");
    LOG_ERROR("To FD 1");
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
} // namespace holovibes::units
