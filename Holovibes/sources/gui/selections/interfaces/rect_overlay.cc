#include "rect_overlay.hh"
#include "BasicOpenGLWindow.hh"

#include "rect_gl.hh"

namespace holovibes::gui
{
RectOverlay::RectOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent)
    : Overlay(overlay, parent)
{
    LOG_FUNC();
}

void RectOverlay::init()
{
    // Program_ already bound by caller (initProgram)

    Vao_.bind();

    // Set vertices position (it's a rectangle that fill up the entiere window)
    const float vertices[] = {
        -1.f,
        -1.f,
        1.f,
        -1.f,
        1.f,
        1.f,
        -1.f,
        1.f,
    };
    glGenBuffers(1, &verticesIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(verticesShader_);
    glVertexAttribPointer(verticesShader_, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
    glDisableVertexAttribArray(verticesShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Set color
    const float colorData[] = {
        color_[0],
        color_[1],
        color_[2],
        color_[0],
        color_[1],
        color_[2],
        color_[0],
        color_[1],
        color_[2],
        color_[0],
        color_[1],
        color_[2],
    };
    glGenBuffers(1, &colorIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, colorIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(colorShader_);
    glVertexAttribPointer(colorShader_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glDisableVertexAttribArray(colorShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Set vertices order
    const GLuint elements[] = {0, 1, 1, 2, 2, 3, 3, 0};
    glGenBuffers(1, &elemIndex_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    Vao_.release();

    // Program_ released by caller (initProgram)
}

void RectOverlay::draw()
{
    initDraw();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glDrawElements(GL_LINES, 8, GL_UNSIGNED_INT, nullptr);

    endDraw();
}

void RectOverlay::setBuffer()
{
    // Normalizing the zone to (-1; 1)
    RectGL zone_gl(*parent_, zone_);

    // The translation is the center of the rectangle
    translation_.x = (zone_gl.x() + zone_gl.right()) / 2;
    translation_.y = (zone_gl.y() + zone_gl.bottom()) / 2;

    // The scale is the size of the rectangle divide by 2 (because if horizontal it will scale left and also right)
    scale_.x = (zone_gl.right() - zone_gl.x()) / 2;
    scale_.y = (zone_gl.bottom() - zone_gl.y()) / 2;
}

void RectOverlay::move(QMouseEvent* e)
{
    if (e->buttons() == Qt::LeftButton)
    {
        auto pos = getMousePos(e->pos());
        zone_.set_dst(pos);
        checkCorners();
        setBuffer();
        display_ = true;
    }
}

void RectOverlay::checkCorners()
{
    auto parent_fd = parent_->getFd();

    if (zone_.dst().x() < 0)
        zone_.dst_ref().x() = 0;
    else if (zone_.dst().x() > parent_fd.width)
        zone_.dst_ref().x() = parent_fd.width;

    if (zone_.dst().y() < 0)
        zone_.dst_ref().y() = 0;
    else if (zone_.dst().y() > parent_fd.height)
        zone_.dst_ref().y() = parent_fd.height;
}
} // namespace holovibes::gui
