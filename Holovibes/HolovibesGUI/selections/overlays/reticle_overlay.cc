#include "API.hh"
#include "reticle_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
ReticleOverlay::ReticleOverlay(BasicOpenGLWindow* parent)
    : Overlay(KindOfOverlay::Reticle, parent)
{
    LOG_FUNC();

    display_ = true;
    alpha_ = 1.0f;
}

void ReticleOverlay::init() { setBuffer(); }

void ReticleOverlay::draw()
{
    parent_->makeCurrent();
    setBuffer();
    Vao_.bind();
    Program_->bind();

    glEnableVertexAttribArray(colorShader_);
    glEnableVertexAttribArray(verticesShader_);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    Program_->setUniformValue(Program_->uniformLocation("alpha"), alpha_);
    glDrawElements(GL_LINES, 12, GL_UNSIGNED_INT, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableVertexAttribArray(verticesShader_);
    glDisableVertexAttribArray(colorShader_);

    Program_->release();
    Vao_.release();
}

void ReticleOverlay::setBuffer()
{
    Program_->bind();
    Vao_.bind();
    float scale = api::get_reticle().scale;
    float w = parent_->size().width();
    float h = parent_->size().height();
    float w_2 = w / 2;
    float h_2 = h / 2;
    // float cross_width = 20.0f / w_2;
    // float cross_height = 20.0f / h_2;
    float w_border = (w_2 * scale) / w_2;
    float h_border = (h_2 * scale) / h_2;

    units::ConversionData convert(parent_);
    auto top_left = units::PointWindow(convert, w_2 - w_2 * scale, h_2 - h_2 * scale);
    auto bottom_right = units::PointWindow(convert, w_2 + w_2 * scale, h_2 + h_2 * scale);
    units::RectWindow zone_window(top_left, bottom_right);
    zone_ = zone_window;
    api::set_reticle_zone(zone_);

    /*
            0-------------1
            |             |
            |      4      |
            |    6-+-7    |
            |      5      |
            |             |
            3-------------2
    */

    const float vertices[] = {
        -w_border,
        -h_border,
        w_border,
        -h_border,
        w_border,
        h_border,
        -w_border,
        h_border,
    };
    glGenBuffers(1, &verticesIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(verticesShader_);
    glVertexAttribPointer(verticesShader_, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
    glDisableVertexAttribArray(verticesShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    const float colorData[] = {
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
    };
    glGenBuffers(1, &colorIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, colorIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colorData), colorData, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(colorShader_);
    glVertexAttribPointer(colorShader_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glDisableVertexAttribArray(colorShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    const GLuint elements[] = {
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        0,
    };
    glGenBuffers(1, &elemIndex_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    Vao_.release();
    Program_->release();
}
} // namespace holovibes::gui
