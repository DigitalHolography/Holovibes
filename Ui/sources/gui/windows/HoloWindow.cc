#include <sstream>

#include "API.hh"
#include "HoloWindow.hh"
#include "MainWindow.hh"
#include "SliceWindow.hh"
#include "tools.hh"
#include "API.hh"
#include "GUI.hh"

namespace holovibes::gui
{
HoloWindow::HoloWindow(
    QPoint p, QSize s, DisplayQueue* q, std::unique_ptr<SliceWindow>& xz, std::unique_ptr<SliceWindow>& yz, float ratio)
    : RawWindow(p, s, q, ratio, KindOfView::Hologram)
    , xz_slice_(xz)
    , yz_slice_(yz)
{
}

HoloWindow::~HoloWindow() {}

void HoloWindow::initShaders()
{
    Program = new QOpenGLShaderProgram();
    Program->addShaderFromSourceFile(
        QOpenGLShader::Vertex,
        gui::create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / "vertex.holo.glsl").string()));
    Program->addShaderFromSourceFile(
        QOpenGLShader::Fragment,
        gui::create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / "fragment.tex.glsl").string()));
    Program->link();

    overlay_manager_.create_default();
}

void HoloWindow::focusInEvent(QFocusEvent* e)
{
    QOpenGLWindow::focusInEvent(e);
    API.view.change_window(WindowKind::XYview);
    NotifierManager::notify("notify", true);
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
