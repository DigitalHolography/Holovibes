#include "Overlay.hh"
#include "BasicOpenGLWindow.hh"
#include "RawWindow.hh"
#include "HoloWindow.hh"
#include "logger.hh"
#include "tools.hh"

namespace holovibes
{
namespace gui
{
Overlay::Overlay(KindOfOverlay overlay, BasicOpenGLWindow* parent)
    : QOpenGLFunctions()
    , zone_(units::ConversionData(parent))
    , kOverlay_(overlay)
    , verticesIndex_(0)
    , colorIndex_(0)
    , elemIndex_(0)
    , alpha_(0.7f)
    , active_(true)
    , display_(false)
    , parent_(parent)
    , verticesShader_(2)
    , colorShader_(3)
{
}

Overlay::~Overlay()
{
    parent_->makeCurrent();
    glDeleteBuffers(1, &elemIndex_);
    glDeleteBuffers(1, &verticesIndex_);
    glDeleteBuffers(1, &colorIndex_);
}

const units::RectFd& Overlay::getZone() const { return zone_; }

const KindOfOverlay Overlay::getKind() const { return kOverlay_; }

const bool Overlay::isDisplayed() const { return display_; }

const bool Overlay::isActive() const { return active_; }

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
        auto pos = getMousePos(e->pos());
        zone_.setSrc(pos);
        zone_.setDst(zone_.src());
    }
}

void Overlay::keyPress(QKeyEvent*) {}

void Overlay::initProgram()
{
    parent_->makeCurrent();
    initializeOpenGLFunctions();
    Program_ = std::make_unique<QOpenGLShaderProgram>();
    Program_->addShaderFromSourceFile(
        QOpenGLShader::Vertex,
        create_absolute_qt_path("shaders/vertex.overlay.glsl"));
    Program_->addShaderFromSourceFile(
        QOpenGLShader::Fragment,
        create_absolute_qt_path("shaders/fragment.color.glsl"));
    Vao_.create();
    if (!Program_->bind())
        LOG_ERROR(Program_->log().toStdString());
    init();
    Program_->release();
}

units::PointWindow Overlay::getMousePos(const QPoint& pos)
{
    auto x = pos.x();
    auto y = pos.y();
    units::PointWindow res(units::ConversionData(parent_), x, y);
    return res;
}

void Overlay::print()
{
    std::cout << "Kind: " << kOverlay_ << ", zone: " << zone_
              << ", active: " << active_ << ", display: " << display_
              << std::endl;
}
} // namespace gui
} // namespace holovibes
