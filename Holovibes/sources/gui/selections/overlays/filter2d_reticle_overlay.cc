/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "filter2d_reticle_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
namespace gui
{
Filter2DReticleOverlay::Filter2DReticleOverlay(BasicOpenGLWindow* parent)
    : Overlay(Filter2DReticle, parent)
{
    display_ = true;
    alpha_ = 1.0f;
}

void Filter2DReticleOverlay::init() { setBuffer(); }

void Filter2DReticleOverlay::draw()
{
    parent_->makeCurrent();
    setBuffer();
    Vao_.bind();
    Program_->bind();

    glEnableVertexAttribArray(colorShader_);
    glEnableVertexAttribArray(verticesShader_);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    Program_->setUniformValue(Program_->uniformLocation("alpha"), alpha_);
    glDrawElements(GL_LINES, 16, GL_UNSIGNED_INT, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableVertexAttribArray(verticesShader_);
    glDisableVertexAttribArray(colorShader_);
    
    Program_->release();
    Vao_.release();
}

void Filter2DReticleOverlay::setBuffer()
{
    Program_->bind();
    Vao_.bind();
    float w = parent_->size().width();
    float h = parent_->size().height();
    float dimension_min = fmin(w, h);
    float scale_n1 = (parent_->getCd()->filter2d_n1.load() * 2) / dimension_min;
    float scale_n2 = (parent_->getCd()->filter2d_n2.load() * 2) / dimension_min;
    float dim_min_2 = dimension_min / 2;
    float w_border_n1 = (dim_min_2 * scale_n1) / dim_min_2;
    float h_border_n1 = (dim_min_2 * scale_n1) / dim_min_2;
    float w_border_n2 = (dim_min_2 * scale_n2) / dim_min_2;
    float h_border_n2 = (dim_min_2 * scale_n2) / dim_min_2;

    /*
            0-------------1
            |             |
            |    4---5    |
            |    |   |    |
            |    7---6    |
            |             |
            3-------------2
    */

    const float vertices[] = {
        -w_border_n1, -h_border_n1,
        w_border_n1, -h_border_n1,
        w_border_n1, h_border_n1,
        -w_border_n1, h_border_n1,

        -w_border_n2, -h_border_n2,
        w_border_n2, -h_border_n2,
        w_border_n2, h_border_n2,
        -w_border_n2, h_border_n2,
    };
    glGenBuffers(1, &verticesIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(verticesShader_);
    glVertexAttribPointer(verticesShader_,
                          2,
                          GL_FLOAT,
                          GL_FALSE,
                          2 * sizeof(float),
                          0);
    glDisableVertexAttribArray(verticesShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    const float colorData[] = {
        1, 1, 1,
        1, 1, 0,
        1, 0, 1,
        1, 0, 0,
        0, 1, 1,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0,
    };
    glGenBuffers(1, &colorIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, colorIndex_);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(colorData),
                 colorData,
                 GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(colorShader_);
    glVertexAttribPointer(colorShader_,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          3 * sizeof(float),
                          0);
    glDisableVertexAttribArray(colorShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    const GLuint elements[] = {
        0, 1,
        1, 2,
        2, 3,
        3, 0,

        4, 5,
        5, 6,
        6, 7,
        7, 4,
    };
    glGenBuffers(1, &elemIndex_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(elements),
                 elements,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    Vao_.release();
    Program_->release();
}
} // namespace gui
} // namespace holovibes