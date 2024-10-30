#include "circ_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "API.hh"

#include <cmath>

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
    std::memset(vertices, 0, sizeof(float) * resolution_ * 2);

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
    // trigger basicopenglwindow painwindow() dynamically
    parent_->makeCurrent();

    setBuffer();

    Vao_.bind();
    Program_->bind();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glEnableVertexAttribArray(colorShader_);
    glEnableVertexAttribArray(verticesShader_);

    setUniform();

    glDrawElements(GL_LINES, resolution_ * 2, GL_UNSIGNED_INT, nullptr);

    glDisableVertexAttribArray(verticesShader_);
    glDisableVertexAttribArray(colorShader_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    Program_->release();
    Vao_.release();
}

void CircOverlay::setBuffer()
{
    parent_->makeCurrent();
    Program_->bind();

    // Normalizing the zone to (-1; 1)
    units::RectOpengl zone_gl = zone_;

    float* subVertices = new float[resolution_ * 2];
    for (uint i = 0; i < resolution_; ++i)
    {
        float angle = 2.0f * M_PI * i / resolution_;
        subVertices[i * 2] = zone_gl.src().x() + radius_ * cos(angle);     // x-coordinate
        subVertices[i * 2 + 1] = zone_gl.src().y() + radius_ * sin(angle); // y-coordinate
    }

    // Updating the buffer at verticesIndex_ with new coordinates
    glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * resolution_ * 2, subVertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    Program_->release();
    delete subVertices;
}

void CircOverlay::checkBounds()
{
    auto parent_fd = parent_->getFd();

    if (zone_.src().x() < 0)
        zone_.srcRef().x().set(0);
    if (zone_.src().x() > parent_fd.width)
        zone_.srcRef().x().set(parent_fd.width);

    if (zone_.src().y() < 0)
        zone_.srcRef().y().set(0);
    if (zone_.src().y() > parent_fd.height)
        zone_.srcRef().y().set(parent_fd.height);
}
} // namespace holovibes::gui
