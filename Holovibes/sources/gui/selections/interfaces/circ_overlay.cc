#include "circ_overlay.hh"

#include <cmath>

#include "API.hh"
#include "BasicOpenGLWindow.hh"
#include "rect_gl.hh"

namespace holovibes::gui
{
CircOverlay::CircOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent, uint resolution)
    : Overlay(overlay, parent)
    , resolution_(resolution)
    , radius_(0.5f)
{
    LOG_FUNC();
}

void CircOverlay::init()
{
    // Program_ already bound by caller (initProgram)

    Vao_.bind();

    // Set vertices position
    float* vertices = new float[resolution_ * 2];
    for (uint i = 0; i < resolution_; ++i)
    {
        float angle = 2.0f * M_PI * i / resolution_;
        vertices[i * 2] = cos(angle);     // x-coordinate
        vertices[i * 2 + 1] = sin(angle); // y-coordinate
    }

    glGenBuffers(1, &verticesIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * resolution_ * 2, vertices, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(verticesShader_);
    glVertexAttribPointer(verticesShader_, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
    glDisableVertexAttribArray(verticesShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Set color
    float* colorData = new float[resolution_ * 3];
    for (uint i = 0; i < resolution_; ++i)
    {
        colorData[i * 3] = color_[0];
        colorData[i * 3 + 1] = color_[1];
        colorData[i * 3 + 2] = color_[2];
    }

    glGenBuffers(1, &colorIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, colorIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * resolution_ * 3, colorData, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(colorShader_);
    glVertexAttribPointer(colorShader_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glDisableVertexAttribArray(colorShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Set vertices order
    GLuint* elements = new GLuint[resolution_ * 2];

    for (uint i = 0; i < resolution_; ++i)
    {
        elements[i * 2] = i;
        elements[i * 2 + 1] = (i + 1) % resolution_;
    }

    glGenBuffers(1, &elemIndex_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * resolution_ * 2, elements, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    Vao_.release();

    delete vertices;
    delete colorData;
    delete elements;

    // Program_ released by caller (initProgram)
}

void CircOverlay::draw()
{
    initDraw();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glDrawElements(GL_LINES, resolution_ * 2, GL_UNSIGNED_INT, nullptr);

    endDraw();
}

void CircOverlay::setBuffer()
{
    // Normalizing the zone to (-1; 1)
    RectGL zone_gl(*parent_, zone_);

    // The translation is the center of the rectangle
    translation_.x = (zone_gl.x() + zone_gl.right()) / 2;
    translation_.y = (zone_gl.y() + zone_gl.bottom()) / 2;

    scale_.x = radius_;
    scale_.y = radius_;
}

void CircOverlay::checkBounds()
{
    auto parent_fd = parent_->getFd();

    if (zone_.src().x() < 0)
        zone_.srcRef().x() = 0;
    if (zone_.src().x() > parent_fd.width)
        zone_.srcRef().x() = parent_fd.width;

    if (zone_.src().y() < 0)
        zone_.srcRef().y() = 0;
    if (zone_.src().y() > parent_fd.height)
        zone_.srcRef().y() = parent_fd.height;
}
} // namespace holovibes::gui
