#include "Overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "RawWindow.hh"
#include "HoloWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "GUI.hh"

namespace holovibes::gui
{
Overlay::Overlay(KindOfOverlay overlay, BasicOpenGLWindow* parent)
    : QOpenGLFunctions()
    , zone_()
    , kOverlay_(overlay)
    , verticesIndex_(0)
    , colorIndex_(0)
    , elemIndex_(0)
    , verticesShader_(2)
    , colorShader_(3)
    , alpha_(1.f)
    , color_({1.f, 0.f, 0.f})
    , active_(true)
    , display_(false)
    , parent_(parent)
    , scale_(1.f)
    , translation_(0.f)
{
    LOG_FUNC();
}

Overlay::~Overlay()
{
    parent_->makeCurrent();
    glDeleteBuffers(1, &elemIndex_);
    glDeleteBuffers(1, &verticesIndex_);
    glDeleteBuffers(1, &colorIndex_);
}

void Overlay::onSetCurrent()
{
    // Do nothing
}

void Overlay::disable()
{
    active_ = false;
    display_ = false;
}

void Overlay::enable()
{
    active_ = true;
    display_ = true;
}

void Overlay::press(QMouseEvent* e)
{
    if (e->button() == Qt::LeftButton)
    {
        units::PointFd pos = getMousePos(e->pos());
        zone_.setSrc(pos);
        zone_.setDst(zone_.src());
    }
}

void Overlay::keyPress(QKeyEvent*) {}

void Overlay::initProgram()
{
    LOG_FUNC();
    parent_->makeCurrent();
    initializeOpenGLFunctions();
    Program_ = std::make_unique<QOpenGLShaderProgram>();
    Program_->addShaderFromSourceFile(
        QOpenGLShader::Vertex,
        gui::create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / "vertex.overlay.glsl").string()));
    Program_->addShaderFromSourceFile(
        QOpenGLShader::Fragment,
        gui::create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / "fragment.color.glsl").string()));
    Vao_.create();
    // if (!Program_->bind())
    //     LOG_ERROR("Shader error : {}", Program_->log().toStdString());
    init();
    Program_->release();
}

void Overlay::initDraw()
{
    parent_->makeCurrent();

    setBuffer();

    Vao_.bind();

    // Bind program and set uniform
    Program_->bind();
    Program_->setUniformValue(Program_->uniformLocation("alpha"), alpha_);
    Program_->setUniformValue(Program_->uniformLocation("scale"), scale_.x, scale_.y);
    Program_->setUniformValue(Program_->uniformLocation("translation"), translation_.x, translation_.y);

    glEnableVertexAttribArray(colorShader_);
    glEnableVertexAttribArray(verticesShader_);
}

void Overlay::endDraw()
{
    glDisableVertexAttribArray(verticesShader_);
    glDisableVertexAttribArray(colorShader_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    Program_->release();
    Vao_.release();
}

units::PointFd Overlay::getMousePos(const QPoint& pos)
{
    float x = pos.x();
    float y = pos.y();

    x = (x / parent_->width()) * parent_->getFd().width;
    y = (y / parent_->height()) * parent_->getFd().height;

    units::PointFd res(x, y);
    return res;
}

void Overlay::print()
{
    std::ostringstream zone_oss;
    zone_oss << zone_;
    LOG_INFO("Kind: {}, zone: {}, active: {}, display: {}", (int)kOverlay_, zone_oss.str(), active_, display_);
}
} // namespace holovibes::gui
