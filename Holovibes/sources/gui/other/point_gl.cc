#include "point_gl.hh"

#include "API.hh"

namespace holovibes::gui
{

PointGL::PointGL(const BasicOpenGLWindow& window, units::PointFd point)
{
    float x = point.x();
    float y = point.y();

    float fd_width = window.getFd().width;
    float fd_height = window.getFd().height;

    x_ = 2.f * (x / fd_width) - 1.f;
    y_ = - (2.f * (y / fd_height) - 1.f);

    // By matrix multiplication
    glm::vec3 input{x_, y_, 1.0f};
    const auto& matrix = window.getTransformMatrix();
    auto output = matrix * input;
    x_ = output[0];
    y_ = output[1];
}

units::PointFd PointGL::to_fd() const
{
    float x = (x_ + 1.f) / 2.f;
    x *= API.input.get_fd().width;

    float y = (-y_ + 1.f) / 2.f;
    y *= API.input.get_fd().height;

    units::PointFd res(x, y);

    return res;
}
} // namespace holovibes::gui
