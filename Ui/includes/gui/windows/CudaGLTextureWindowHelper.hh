#pragma once

#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QWheelEvent>
#include <QFocusEvent>
#include <QCloseEvent>
#include <cuda_gl_interop.h>

#include "CudaTexture.hh"
#include "display_queue.hh"
#include "overlay_manager.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
class CudaGLTextureWindowHelper : public BasicOpenGLWindow
{
  public:
    CudaGLTextureWindowHelper(QPoint p, QSize s, DisplayQueue* q, KindOfView k);
    virtual ~CudaGLTextureWindowHelper();

    void initializeGL();
    void initShaders();
    void paintGL();

  protected:
    // OpenGL/CUDA resources
    CudaTexture* cudaTexture = nullptr;
};
} // namespace holovibes::gui