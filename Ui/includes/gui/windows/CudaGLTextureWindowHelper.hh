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

    // Initialization
    void initializeGL( const std::string vertex_shader_path,  const std::string fragment_shader_path);
    void initShaders( const std::string vertex_shader_path,  const std::string fragment_shader_path);

    // Rendering
    void paintGL(void* frame);

    // void cleanup();

    // // Accessors
    // GLuint getTextureID() const;

protected:
    // OpenGL/CUDA resources
    CudaTexture* cudaTexture = nullptr;

};
}