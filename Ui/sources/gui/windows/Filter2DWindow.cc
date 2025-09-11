// Windows include is needed for the cuda_gl_interop header to compile
#ifdef WIN32
#include <windows.h>
#endif
#include <cuda_gl_interop.h>

#include "texture_update.cuh"
#include "Filter2DWindow.hh"
#include "MainWindow.hh"
#include "tools.hh"
#include "API.hh"
#include "GUI.hh"
#include "user_interface_descriptor.hh"

namespace holovibes::gui
{
Filter2DWindow::Filter2DWindow(QPoint p, QSize s, DisplayQueue* q)
    : CudaGLTextureWindowHelper(p, s, q, KindOfView::Filter2D)
{
    LOG_FUNC();

    setMinimumSize(s);
    show();
}

Filter2DWindow::~Filter2DWindow()
{
#ifdef NDEBUG
    if (cuResource)
    {
        cudaSafeCall(cudaGraphicsUnmapResources(1, &cuResource, cuStream));
        cudaSafeCall(cudaGraphicsUnregisterResource(cuResource));
    }
#endif
}

void Filter2DWindow::focusInEvent(QFocusEvent* e)
{
    QWindow::focusInEvent(e);
    API.view.change_window(WindowKind::Filter2D);
    NotifierManager::notify("notify", true);
}

void Filter2DWindow::closeEvent(QCloseEvent* e)
{
    API.view.set_filter2d_view(false);
    gui::set_filter2d_view(false, 0);
    NotifierManager::notify("notify", true);
}
} // namespace holovibes::gui
