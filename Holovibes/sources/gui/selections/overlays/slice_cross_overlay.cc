#include "API.hh"
#include "slice_cross_overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "API.hh"

namespace holovibes::gui
{
SliceCrossOverlay::SliceCrossOverlay(BasicOpenGLWindow* parent)
    : RectOverlay(KindOfOverlay::SliceCross, parent)
    , line_alpha_(0.5f)
    , elemLineIndex_(0)
    , locked_(true)
    , pIndex_(0, 0)
{
    color_ = {1.f, 0.f, 0.f};
    alpha_ = 0.05f;
    display_ = true;
}

SliceCrossOverlay::~SliceCrossOverlay()
{
    parent_->makeCurrent();
    glDeleteBuffers(1, &elemLineIndex_);
}

void SliceCrossOverlay::init()
{
    RectOverlay::init();

    // Set line vertices order
    const GLuint elements[] = {0, 1, 1, 2, 2, 3, 3, 0};

    glGenBuffers(1, &elemLineIndex_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void SliceCrossOverlay::draw()
{
    parent_->makeCurrent();
    setBuffer();
    Vao_.bind();
    Program_->bind();

    glEnableVertexAttribArray(colorShader_);
    glEnableVertexAttribArray(verticesShader_);

    // Drawing two lines
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
    Program_->setUniformValue(Program_->uniformLocation("alpha"), line_alpha_);
    glDrawElements(GL_LINES, 8, GL_UNSIGNED_INT, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Drawing area between two lines
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    Program_->setUniformValue(Program_->uniformLocation("alpha"), alpha_);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableVertexAttribArray(verticesShader_);
    glDisableVertexAttribArray(colorShader_);

    Program_->release();
    Vao_.release();
}

void SliceCrossOverlay::keyPress(QKeyEvent* e)
{
    if (e->key() == Qt::Key_Space)
    {
        locked_ = !locked_;
        parent_->setCursor(locked_ ? Qt::ArrowCursor : Qt::CrossCursor);
    }
}

void SliceCrossOverlay::move(QMouseEvent* e)
{
    if (!locked_)
    {
        auto kView = parent_->getKindOfView();

        pIndex_ = getMousePos(e->pos());

        uint p = (kView == KindOfView::SliceXZ) ? pIndex_.y() : pIndex_.x();
        api::set_p_index(p);
    }
}

void SliceCrossOverlay::release(ushort frameside) {}

void SliceCrossOverlay::setBuffer()
{
    units::PointFd topLeft;
    units::PointFd bottomRight;
    auto kView = parent_->getKindOfView();

    ViewPQ p = api::get_p();

    uint pmin = p.start;
    uint pmax = pmin + p.width;

    units::ConversionData convert(parent_);

    pmax = (pmax + 1);
    topLeft = (kView == KindOfView::SliceXZ) ? units::PointFd(convert, 0, pmin) : units::PointFd(convert, pmin, 0);
    bottomRight = (kView == KindOfView::SliceXZ) ? units::PointFd(convert, parent_->getFd().width, pmax)
                                                 : units::PointFd(convert, pmax, parent_->getFd().height);
    zone_ = units::RectFd(topLeft, bottomRight);

    // Updating opengl buffer
    RectOverlay::setBuffer();
}
} // namespace holovibes::gui
