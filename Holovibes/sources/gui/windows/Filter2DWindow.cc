/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include <sstream>

#include "Filter2DWindow.hh"
#include "MainWindow.hh"
#include "SliceWindow.hh"
#include "tools.hh"

namespace holovibes
{
namespace gui
{
Filter2DWindow::Filter2DWindow(QPoint p,
                       QSize s,
                       DisplayQueue* q,
                       SharedPipe ic,
                       MainWindow* main_window)
    : RawWindow(p, s, q, KindOfView::Hologram)
    , Ic(ic)
    , main_window_(main_window)
{
}

Filter2DWindow::~Filter2DWindow() {}

std::shared_ptr<ICompute> Filter2DWindow::getPipe() { return Ic; }

void Filter2DWindow::initShaders()
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

void Filter2DWindow::focusInEvent(QFocusEvent* e)
{
    QOpenGLWindow::focusInEvent(e);
    cd_->current_window = WindowKind::Filter2D;
    cd_->notify_observers();
}

void Filter2DWindow::resetTransform() { BasicOpenGLWindow::resetTransform(); }

void Filter2DWindow::setTransform()
{
    BasicOpenGLWindow::setTransform();
}
} // namespace gui
} // namespace holovibes
