#include "cross_overlay.hh"

#include <sstream>

#include "API.hh"
#include "BasicOpenGLWindow.hh"
#include "holovibes.hh"
#include "notifier.hh"
#include "rect_gl.hh"

namespace holovibes::gui
{
CrossOverlay::CrossOverlay(BasicOpenGLWindow* parent)
    : Overlay(KindOfOverlay::Cross, parent)
    , line_alpha_(0.5f)
    , elemLineIndex_(0)
    , locked_(true)
{
    LOG_FUNC();

    color_ = {1.f, 0.f, 0.f};
    alpha_ = 0.1f;
    display_ = true;
}

CrossOverlay::~CrossOverlay()
{
    parent_->makeCurrent();
    glDeleteBuffers(1, &elemLineIndex_);
}

#define VERTEX_COUNT 8 // 4 by rectangle

void CrossOverlay::init()
{
    // Program_ already bound by caller (initProgram)
    Vao_.bind();

    // Set vertices position
    const float vertices[VERTEX_COUNT * 2] = {0};

    glGenBuffers(1, &verticesIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * VERTEX_COUNT, vertices, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(verticesShader_);
    glVertexAttribPointer(verticesShader_, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
    glDisableVertexAttribArray(verticesShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Set color
    float colorData[VERTEX_COUNT * 3];
    for (uint i = 0; i < VERTEX_COUNT; ++i)
    {
        colorData[i * 3] = color_[0];
        colorData[i * 3 + 1] = color_[1];
        colorData[i * 3 + 2] = color_[2];
    }

    glGenBuffers(1, &colorIndex_);
    glBindBuffer(GL_ARRAY_BUFFER, colorIndex_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * VERTEX_COUNT, colorData, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(colorShader_);
    glVertexAttribPointer(colorShader_, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glDisableVertexAttribArray(colorShader_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Set line vertices order
    const GLuint lineElements[] = {// vertical rectangle
                                   0,
                                   1,
                                   1,
                                   2,
                                   2,
                                   3,
                                   3,
                                   0,
                                   // horizontal rectangle
                                   4,
                                   5,
                                   5,
                                   6,
                                   6,
                                   7,
                                   7,
                                   4};

    glGenBuffers(1, &elemLineIndex_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(lineElements), lineElements, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Set rectangle vertices order
    std::vector<GLuint> elements{// vertical area
                                 0,
                                 1,
                                 2,
                                 2,
                                 3,
                                 0,
                                 // horizontal area
                                 4,
                                 5,
                                 6,
                                 6,
                                 7,
                                 4};

    glGenBuffers(1, &elemIndex_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, elements.size() * sizeof(GLuint), elements.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    Vao_.release();
    // Program_ released by caller (initProgram)
}

void CrossOverlay::draw()
{
    initDraw();

    // Drawing areas between lines
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Drawing four lines
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
    Program_->setUniformValue(Program_->uniformLocation("alpha"), line_alpha_);
    glDrawElements(GL_LINES, 16, GL_UNSIGNED_INT, nullptr);

    endDraw();
}

void CrossOverlay::onSetCurrent()
{
    mouse_position_ = units::PointFd(API.transform.get_x_cuts(), API.transform.get_y_cuts());
}

void CrossOverlay::press(QMouseEvent* e) {}

void CrossOverlay::keyPress(QKeyEvent* e)
{
    if (e->key() == Qt::Key_Space)
    {
        locked_ = !locked_;
        parent_->setCursor(locked_ ? Qt::ArrowCursor : Qt::CrossCursor);
    }
}

void CrossOverlay::move(QMouseEvent* e)
{
    if (!locked_)
    {
        units::PointFd pos = getMousePos(e->pos());
        mouse_position_ = pos;

        API.transform.set_x_cuts(mouse_position_.x());
        API.transform.set_y_cuts(mouse_position_.y());

        NotifierManager::notify("notify", true);
    }
}

void CrossOverlay::release(ushort frameside) {}

void CrossOverlay::computeZone()
{
    units::PointFd topLeft;
    units::PointFd bottomRight;

    // Computing min/max coordinates in function of the frame_descriptor
    int x_start = API.transform.get_x_cuts();
    int x_width = API.transform.get_x_accu_level();
    int y_start = API.transform.get_y_cuts();
    int y_width = API.transform.get_y_accu_level();

    int x_min = x_start;
    int x_max = x_start;
    int y_min = y_start;
    int y_max = y_start;
    (x_width < 0 ? x_min : x_max) += x_width;
    (y_width < 0 ? y_min : y_max) += y_width;

    units::PointFd min(x_min, y_min);
    units::PointFd max(x_max, y_max);

    // Setting the zone_
    if (x_width == 0)
    {
        min.x() = x_start;
        max.x() = x_start;
    }
    if (y_width == 0)
    {
        min.y() = y_start;
        max.y() = y_start;
    }
    max.x() += 1;
    max.y() += 1;
    zone_ = units::RectFd(min.x(), 0, max.x(), parent_->getFd().height);
    horizontal_zone_ = units::RectFd(0, min.y(), parent_->getFd().width, max.y());
}

void CrossOverlay::setBuffer()
{
    computeZone();

    parent_->makeCurrent();
    Program_->bind();

    const RectGL zone_gl(*parent_, zone_);
    const RectGL h_zone_gl(*parent_, horizontal_zone_);

    const float subVertices[] = {zone_gl.x(),
                                 zone_gl.y(),
                                 zone_gl.right(),
                                 zone_gl.y(),
                                 zone_gl.right(),
                                 zone_gl.bottom(),
                                 zone_gl.x(),
                                 zone_gl.bottom(),

                                 h_zone_gl.x(),
                                 h_zone_gl.y(),
                                 h_zone_gl.right(),
                                 h_zone_gl.y(),
                                 h_zone_gl.right(),
                                 h_zone_gl.bottom(),
                                 h_zone_gl.x(),
                                 h_zone_gl.bottom()};

    // Updating the buffer at verticesIndex_ with new coordinates
    glBindBuffer(GL_ARRAY_BUFFER, verticesIndex_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(subVertices), subVertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    Program_->release();
}

} // namespace holovibes::gui
