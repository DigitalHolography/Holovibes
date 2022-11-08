// Windows include is needed for the cuda_gl_interop header to compile
#ifdef WIN32
#include <windows.h>
#endif
#include <cuda_gl_interop.h>

#include "API.hh"
#include "texture_update.cuh"
#include "Filter2DWindow.hh"
#include "MainWindow.hh"
#include "tools.hh"
#include "API.hh"

namespace holovibes::gui
{
Filter2DWindow::Filter2DWindow(QPoint p, QSize s, DisplayQueue* q)
    : BasicOpenGLWindow(p, s, q, KindOfView::ViewFilter2D)
{
    LOG_FUNC();

    setMinimumSize(s);
    show();
}

Filter2DWindow::~Filter2DWindow()
{
    if (cuResource)
    {
        cudaSafeCall(cudaGraphicsUnmapResources(1, &cuResource, cuStream));
        cudaSafeCall(cudaGraphicsUnregisterResource(cuResource));
    }
}

void Filter2DWindow::initShaders()
{
    Program = new QOpenGLShaderProgram();
    Program->addShaderFromSourceFile(QOpenGLShader::Vertex, create_absolute_qt_path("shaders/vertex.holo.glsl"));
    Program->addShaderFromSourceFile(QOpenGLShader::Fragment, create_absolute_qt_path("shaders/fragment.tex.glsl"));
    Program->link();
    overlay_manager_.create_default();
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
    glGenTextures(1, &Tex);
    glBindTexture(GL_TEXTURE_2D, Tex);

    size_t size = fd_.get_frame_size();
    ushort* mTexture = new ushort[size];
    std::memset(mTexture, 0, size * sizeof(ushort));

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fd_.width, fd_.height, 0, GL_RG, GL_UNSIGNED_SHORT, mTexture);

    Program->setUniformValue(Program->uniformLocation("tex"), 0);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    GL_NEAREST); // GL_NEAREST ~ GL_LINEAR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    if (fd_.depth == 8)
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_GREEN);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    delete[] mTexture;
    cudaGraphicsGLRegisterImage(&cuResource,
                                Tex,
                                GL_TEXTURE_2D,
                                cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaGraphicsMapResources(1, &cuResource, cuStream);
    cudaGraphicsSubResourceGetMappedArray(&cuArray, cuResource, 0, 0);
    cuArrRD.resType = cudaResourceTypeArray;
    cuArrRD.res.array.array = cuArray;
    cudaCreateSurfaceObject(&cuSurface, &cuArrRD);
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
    startTimer(1000 / api::get_display_rate());
}

void Filter2DWindow::paintGL()
{
    makeCurrent();
    glClear(GL_COLOR_BUFFER_BIT);
    Vao.bind();
    Program->bind();

    textureUpdate(cuSurface, output_->get_last_image(), output_->get_fd(), cuStream);

    glBindTexture(GL_TEXTURE_2D, Tex);
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
    api::set_current_view_kind(WindowKind::ViewFilter2D);
}
} // namespace holovibes::gui
