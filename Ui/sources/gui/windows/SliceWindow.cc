// Windows include is needed for the cuda_gl_interop header to compile
#ifdef WIN32
#include <windows.h>
#endif
#include <cuda_gl_interop.h>

#include "API.hh"
#include "texture_update.cuh"
#include "SliceWindow.hh"
#include "MainWindow.hh"
#include "tools.hh"
#include "GUI.hh"
#include "user_interface_descriptor.hh"

namespace holovibes::gui
{
SliceWindow::SliceWindow(QPoint p, QSize s, DisplayQueue* q, KindOfView k)
    : TextureWindowHelper(p, s, q, k)
    , cuArray(nullptr)
    , cuSurface(0)
{
    LOG_FUNC();

    setMinimumSize(s);
    show();
}

SliceWindow::~SliceWindow()
{
    cudaDestroySurfaceObject(cuSurface);
    cudaFreeArray(cuArray);
}

void SliceWindow::initShaders()
{
    TextureWindowHelper::initShaders();
    if (API.compute.get_img_type() == ImgType::Composite)
        overlay_manager_.enable<Rainbow>();
    else
        overlay_manager_.create_default();
}

void SliceWindow::mousePressEvent(QMouseEvent* e) { overlay_manager_.press(e); }

void SliceWindow::mouseMoveEvent(QMouseEvent* e) { overlay_manager_.move(e); }

void SliceWindow::mouseReleaseEvent(QMouseEvent* e)
{
    overlay_manager_.release(fd_.width);
    if (e->button() == Qt::RightButton)
    {
        resetTransform();
        if (gui::get_main_display())
            gui::get_main_display()->resetTransform();
    }
}

void SliceWindow::focusInEvent(QFocusEvent* e)
{
    QWindow::focusInEvent(e);
    API.view.change_window(kView == KindOfView::SliceXZ ? WindowKind::XZview : WindowKind::YZview);
    NotifierManager::notify("notify", true);
}

void SliceWindow::closeEvent(QCloseEvent* e)
{
    if (kView == KindOfView::SliceXZ)
        API.window_pp.set_enabled(false, WindowKind::XZview);
    else if (kView == KindOfView::SliceYZ)
        API.window_pp.set_enabled(false, WindowKind::YZview);

    if (!API.window_pp.get_enabled(WindowKind::XZview) && !API.window_pp.get_enabled(WindowKind::YZview))
    {
        API.view.set_3d_cuts_view(false);
        gui::set_3d_cuts_view(false, 0);
        NotifierManager::notify("notify", true);
    }
}
} // namespace holovibes::gui
