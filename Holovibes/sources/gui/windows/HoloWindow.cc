#include <sstream>

#include "HoloWindow.hh"
#include "MainWindow.hh"
#include "SliceWindow.hh"
#include "tools.hh"

namespace holovibes
{
namespace gui
{
HoloWindow::HoloWindow(QPoint p,
                       QSize s,
                       DisplayQueue* q,
                       SharedPipe ic,
                       std::unique_ptr<SliceWindow>& xz,
                       std::unique_ptr<SliceWindow>& yz,
                       MainWindow* main_window)
    : RawWindow(p, s, q, KindOfView::Hologram)
    , Ic(ic)
    , main_window_(main_window)
    , xz_slice_(xz)
    , yz_slice_(yz)
{
}

HoloWindow::~HoloWindow() {}

std::shared_ptr<ICompute> HoloWindow::getPipe() { return Ic; }

void HoloWindow::initShaders()
{
    Program = new QOpenGLShaderProgram();
    Program->addShaderFromSourceFile(
        QOpenGLShader::Vertex,
        create_absolute_qt_path("shaders/vertex.holo.glsl"));
    Program->addShaderFromSourceFile(
        QOpenGLShader::Fragment,
        create_absolute_qt_path("shaders/fragment.tex.glsl"));
    Program->link();
    overlay_manager_.create_default();
}

void HoloWindow::focusInEvent(QFocusEvent* e)
{
    QOpenGLWindow::focusInEvent(e);
    cd_->current_window = WindowKind::XYview;
    cd_->notify_observers();
}

void HoloWindow::update_slice_transforms()
{
    if (xz_slice_)
    {
        xz_slice_->setTranslate(translate_[0], 0);
        xz_slice_->setScale(getScale());
    }
    if (yz_slice_)
    {
        yz_slice_->setTranslate(0, translate_[1]);
        yz_slice_->setScale(getScale());
    }
}

void HoloWindow::resetTransform() { BasicOpenGLWindow::resetTransform(); }

void HoloWindow::setTransform()
{
    BasicOpenGLWindow::setTransform();
    update_slice_transforms();
}
} // namespace gui
} // namespace holovibes
