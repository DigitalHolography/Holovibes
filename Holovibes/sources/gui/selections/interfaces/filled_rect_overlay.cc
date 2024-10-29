#include "filled_rect_overlay.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
FilledRectOverlay::FilledRectOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent)
    : RectOverlay(overlay, parent)
    , fill_elem_index_(0)
    , fill_alpha_(0.05f)
{
    LOG_FUNC();
}

FilledRectOverlay::~FilledRectOverlay()
{
    parent_->makeCurrent();
    glDeleteBuffers(1, &fill_elem_index_);
}

void FilledRectOverlay::init()
{
    RectOverlay::init();

    // Set line vertices order
    const GLuint elements[] = {0, 1, 2, 2, 3, 0};

    glGenBuffers(1, &fill_elem_index_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fill_elem_index_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void FilledRectOverlay::draw()
{
    parent_->makeCurrent();
    setBuffer();
    Vao_.bind();
    Program_->bind();

    glEnableVertexAttribArray(colorShader_);
    glEnableVertexAttribArray(verticesShader_);

    // Drawing two lines
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    Program_->setUniformValue(Program_->uniformLocation("alpha"), alpha_);
    glDrawElements(GL_LINES, 8, GL_UNSIGNED_INT, nullptr);

    // Drawing area between two lines
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fill_elem_index_);
    Program_->setUniformValue(Program_->uniformLocation("alpha"), fill_alpha_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(verticesShader_);
    glDisableVertexAttribArray(colorShader_);

    Program_->release();
    Vao_.release();
}

} // namespace holovibes::gui