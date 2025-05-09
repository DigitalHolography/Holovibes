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
    : BasicOpenGLWindow(p, s, q, KindOfView::Filter2D)
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

void Filter2DWindow::initShaders()
{
    Program = new QOpenGLShaderProgram();
    Program->addShaderFromSourceFile(
        QOpenGLShader::Vertex,
        gui::create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / "vertex.holo.glsl").string()));
    Program->addShaderFromSourceFile(
        QOpenGLShader::Fragment,
        gui::create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / "fragment.tex.glsl").string()));
    Program->link();
}

void Filter2DWindow::initializeGL()
{
    makeCurrent();
    initializeOpenGLFunctions();
    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);

    initShaders();
    Vao.create();
    Vao.bind();
    Program->bind();

#pragma region Texture
    cudaTexture = new CudaTexture(fd_.width, fd_.height, fd_.depth, cuStream);
    if (!cudaTexture->init())
    {
        LOG_ERROR("Failed to initialize CUDA Texture");
    }
#pragma endregion

#pragma region Vertex Buffer Object
    const float data[] = {// Top-left
                          -1.f,
                          1.f, // vertex coord (-1.0f <-> 1.0f)
                          0.0f,
                          0.0f, // texture coord (0.0f <-> 1.0f)
                                // Top-right
                          1.f,
                          1.f,
                          1.f,
                          0.0f,
                          // Bottom-right
                          1.f,
                          -1.f,
                          1.f,
                          1.f,
                          // Bottom-left
                          -1.f,
                          -1.f,
                          0.0f,
                          1.f};
    glGenBuffers(1, &Vbo);
    glBindBuffer(GL_ARRAY_BUFFER, Vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#pragma endregion

#pragma region Element Buffer Object
    const GLuint elements[] = {0, 1, 2, 2, 3, 0};
    glGenBuffers(1, &Ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLuint), elements, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
#pragma endregion

    setTransform();

    Program->release();
    Vao.release();

    glViewport(0, 0, width(), height());
    startTimer(1000 / UserInterfaceDescriptor::instance().display_rate_);
}

void Filter2DWindow::paintGL()
{
    void* frame = output_->get_last_image();
    if (!frame)
        return;

    makeCurrent();
    glClear(GL_COLOR_BUFFER_BIT);
    Vao.bind();
    Program->bind();

    cudaTexture->update(frame, output_->get_fd());

    glBindTexture(GL_TEXTURE_2D, cudaTexture->getTextureID());
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Ebo);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    Program->release();
    Vao.release();

    overlay_manager_.draw();
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
