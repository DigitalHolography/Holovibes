/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include <sstream>

#include "cross_overlay.hh"
#include "BasicOpenGLWindow.hh"

#include "holovibes.hh"

namespace holovibes
{
namespace gui
{
CrossOverlay::CrossOverlay(BasicOpenGLWindow* parent)
    : Overlay(KindOfOverlay::Cross, parent)
    , line_alpha_(0.5f)
    , elemLineIndex_(0)
    , locked_(true)
{
    color_ = {1.f, 0.f, 0.f};
    alpha_ = 0.05f;
    display_ = true;
}

CrossOverlay::~CrossOverlay()
{
    parent_->makeCurrent();
    glDeleteBuffers(1, &elemLineIndex_);
}

void CrossOverlay::init()
{
    // Program_ already bound by caller (initProgram)

    Vao_.bind();

    // Set vertices position
    const float vertices[] = {// vertical area
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              // horizontal area
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f};
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

    // Set color
    const float colorData[] = {// vertical area
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
                               // horizontal area
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
                               color_[2]};
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
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(lineElements),
                 lineElements,
                 GL_STATIC_DRAW);
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
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 elements.size() * sizeof(GLuint),
                 elements.data(),
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    Vao_.release();

    // Program_ released by caller (initProgram)
}

void CrossOverlay::draw()
{
    parent_->makeCurrent();
    computeZone();
    setBuffer();
    Vao_.bind();
    Program_->bind();

    glEnableVertexAttribArray(colorShader_);
    glEnableVertexAttribArray(verticesShader_);

    // Drawing four lines
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemLineIndex_);
    Program_->setUniformValue(Program_->uniformLocation("alpha"), line_alpha_);
    glDrawElements(GL_LINES, 16, GL_UNSIGNED_INT, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Drawing areas between lines
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elemIndex_);
    Program_->setUniformValue(Program_->uniformLocation("alpha"), alpha_);
    glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, nullptr);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableVertexAttribArray(verticesShader_);
    glDisableVertexAttribArray(colorShader_);

    Program_->release();
    Vao_.release();
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

        // Updating infos Tab
        std::stringstream ss;
        ss << "(X,Y) = (" << pos.x() << "," << pos.y() << ")";
        Holovibes::instance().get_info_container().add_indication(
            InformationContainer::IndicationType::CUTS_SLICE_CURSOR,
            ss.str());

        auto cd = parent_->getCd();
        cd->setStftCursor(pos);
        // ---------------
        cd->notify_observers();
    }
}

void CrossOverlay::release(ushort frameside) {}

void CrossOverlay::computeZone()
{
    auto cd = parent_->getCd();
    units::PointFd topLeft;
    units::PointFd bottomRight;
    units::PointFd cursor = cd->getStftCursor();

    // Computing min/max coordinates in function of the frame_descriptor
    units::PointFd cursorPos = cd->getStftCursor();
    int x_min = cursorPos.x();
    int x_max = cursorPos.x();
    int y_min = cursorPos.y();
    int y_max = cursorPos.y();
    if (cd->x_accu_enabled)
        (cd->x_acc_level < 0 ? x_min : x_max) += cd->x_acc_level;
    if (cd->y_accu_enabled)
        (cd->y_acc_level < 0 ? y_min : y_max) += cd->y_acc_level;
    units::ConversionData convert(parent_);
    units::PointFd min(convert, x_min, y_min);
    units::PointFd max(convert, x_max, y_max);

    // Setting the zone_
    if (!cd->x_accu_enabled)
    {
        min.x().set(cursor.x());
        max.x().set(cursor.x());
    }
    if (!cd->y_accu_enabled)
    {
        min.y().set(cursor.y());
        max.y().set(cursor.y());
    }
    max.x() += 1;
    max.y() += 1;
    zone_ =
        units::RectFd(convert, min.x(), 0, max.x(), parent_->getFd().height);
    horizontal_zone_ =
        units::RectFd(convert, 0, min.y(), parent_->getFd().width, max.y());
}

void CrossOverlay::setBuffer()
{
    parent_->makeCurrent();
    Program_->bind();

    const units::RectOpengl zone_gl = zone_;
    const units::RectOpengl h_zone_gl = horizontal_zone_;

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
} // namespace gui
} // namespace holovibes
