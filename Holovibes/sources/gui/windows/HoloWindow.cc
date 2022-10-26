#include <sstream>

#include "API.hh"
#include "HoloWindow.hh"
#include "MainWindow.hh"
#include "SliceWindow.hh"
#include "tools.hh"
#include "API.hh"

namespace holovibes::gui
{
HoloWindow::HoloWindow(
    QPoint p, QSize s, DisplayQueue* q, std::unique_ptr<SliceWindow>& xz, std::unique_ptr<SliceWindow>& yz, float ratio)
    : RawWindow(p, s, q, ratio, KindOfView::Hologram)
    , xz_slice_(xz)
    , yz_slice_(yz)
{
    // FIXME COMPILE WHY
    // if (api::get_current_window().contrast.auto_refresh)
    //     api::get_view_xy().request_exec_auto_contrast();
    api::set_auto_contrast_all();
}

HoloWindow::~HoloWindow() {}

void HoloWindow::initShaders()
{
    Program = new QOpenGLShaderProgram();
    Program->addShaderFromSourceFile(QOpenGLShader::Vertex, create_absolute_qt_path("shaders/vertex.holo.glsl"));
    Program->addShaderFromSourceFile(QOpenGLShader::Fragment, create_absolute_qt_path("shaders/fragment.tex.glsl"));
    Program->link();
    overlay_manager_.create_default();
}

void HoloWindow::focusInEvent(QFocusEvent* e)
{
    QOpenGLWindow::focusInEvent(e);
    api::change_current_window_kind(WindowKind::XYview);
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
} // namespace holovibes::gui
