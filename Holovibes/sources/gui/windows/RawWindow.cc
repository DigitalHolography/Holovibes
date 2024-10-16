#ifdef WIN32
#include <windows.h>
#endif
#include <cuda_gl_interop.h>

#include <QGuiApplication>
#include <QMouseEvent>
#include <QRect>
#include <QScreen>
#include <QWheelEvent>

#include "API.hh"
#include "RawWindow.hh"
#include "HoloWindow.hh"
#include "cuda_memory.cuh"
#include "common.cuh"
#include "tools.hh"
#include "API.hh"

namespace holovibes
{
using camera::Endianness;
using camera::FrameDescriptor;
} // namespace holovibes

namespace holovibes::gui
{
RawWindow::RawWindow(QPoint p, QSize s, DisplayQueue* q, float ratio, KindOfView k)
    : BasicOpenGLWindow(p, s, q, k)
    , texDepth(0)
    , texType(0)
{
    LOG_FUNC();

    this->ratio = ratio;
    show();

    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    int x = json_get_or_default(j_us, 30, "holo window", "x");
    int y = json_get_or_default(j_us, 36, "holo window", "y");
    QPoint point = QPoint(x, y);
    this->setPosition(point);
}

RawWindow::~RawWindow()
{
    // For unknown reasons, this causes a crash in debug and prevents memory leaks in release.
    // It is therefore removed when using the debug mode
#ifdef NDEBUG
    if (cuResource)
        cudaGraphicsUnregisterResource(cuResource);
#endif
}

void RawWindow::initShaders()
{
    Program = new QOpenGLShaderProgram();
    Program->addShaderFromSourceFile(
        QOpenGLShader::Vertex,
        create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / "vertex.raw.glsl").string()));
    Program->addShaderFromSourceFile(
        QOpenGLShader::Fragment,
        create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / "fragment.tex.raw.glsl").string()));
    Program->link();
    overlay_manager_.create_default();
}

void RawWindow::initializeGL()
{
    makeCurrent();
    initializeOpenGLFunctions();
    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);

    initShaders();
    Vao.create();
    Vao.bind();
    Program->bind();

#pragma region Texture
    glGenBuffers(1, &Pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Pbo);
    size_t size;
    if (fd_.depth == camera::PixelDepth::Complex) // cuComplex displayed as a uint
        size = fd_.get_frame_res() * sizeof(uint);
    else if (fd_.depth == camera::PixelDepth::Bits32) // Float are displayed as ushort
        size = fd_.get_frame_res() * sizeof(ushort);
    else
        size = fd_.get_frame_size();

    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr,
                 GL_STATIC_DRAW); // GL_STATIC_DRAW ~ GL_DYNAMIC_DRAW
    glPixelStorei(GL_UNPACK_SWAP_BYTES, (fd_.byteEndian == Endianness::BigEndian) ? GL_TRUE : GL_FALSE);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cuResource, Pbo, cudaGraphicsMapFlags::cudaGraphicsMapFlagsNone);
    /* -------------------------------------------------- */
    glGenTextures(1, &Tex);
    glBindTexture(GL_TEXTURE_2D, Tex);
    texDepth = (fd_.depth == camera::PixelDepth::Bits8) ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT;
    texType = (fd_.depth == camera::PixelDepth::Complex) ? GL_RG : GL_RED;
    if (fd_.depth == camera::PixelDepth::Bits48)
        texType = GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, texType, fd_.width, fd_.height, 0, texType, texDepth, nullptr);

    Program->setUniformValue(Program->uniformLocation("tex"), 0);

    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    GL_NEAREST); // GL_NEAREST ~ GL_LINEAR
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    if (fd_.depth == camera::PixelDepth::Complex)
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_GREEN);
    }
    else if (fd_.depth != camera::PixelDepth::Bits48)
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
#pragma endregion

#pragma region Vertex Buffer Object
    const float data[16] = {// Top-left
                            -1.f,
                            1.f, // vertex coord (-1.0f <-> 1.0f)
                            0.f,
                            0.f, // texture coord (0.0f <-> 1.0f)
                                 // Top-right
                            1.f,
                            1.f,
                            1.f,
                            0.f,
                            // Bottom-right
                            1.f,
                            -1.f,
                            1.f,
                            1.f,
                            // Bottom-left
                            -1.f,
                            -1.f,
                            0.f,
                            1.f};
    glGenBuffers(1, &Vbo);
    glBindBuffer(GL_ARRAY_BUFFER, Vbo);
    glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(float), data, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#pragma endregion

#pragma region Element Buffer Object
    const GLuint elements[6] = {0, 1, 2, 2, 3, 0};
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

/* This part of code makes a resizing of the window displaying image to
   a rectangle format. It also avoids the window to move when resizing.
   There is no visible calling function since it's overriding Qt function.
**/
void RawWindow::resizeGL(int w, int h)
{
    if (ratio == 0.0f)
        return;
    int tmp_width = old_width;
    int tmp_height = old_height;

    auto point = this->position();

    if ((api::get_compute_mode() == Computation::Hologram &&
         api::get_space_transformation() == SpaceTransformation::NONE) ||
        api::get_compute_mode() == Computation::Raw)
    {
        if (w != old_width)
        {
            old_width = w;
            old_height = w / ratio;
        }
        else if (h != old_height)
        {
            old_width = h * ratio;
            old_height = h;
        }
    }
    else
    {
        if (is_resize)
        {
            if (w != old_width)
            {
                old_height = w;
                old_width = w;
            }
            else if (h != old_height)
            {
                old_height = h;
                old_width = h;
            }
        }
        else
        {
            old_height = std::max(h, w);
            old_width = old_height;
        }
        is_resize = true;

        if (old_height < 140 || old_width < 140)
        {
            old_height = tmp_height;
            old_width = tmp_width;
        }
        is_resize = true;
    }

    QRect screen = QGuiApplication::primaryScreen()->geometry();
    if (old_height > screen.height() || old_width > screen.width())
    {
        old_height = tmp_height;
        old_width = tmp_width;
    }
    resize(old_width, old_height);
    this->setPosition(point);
}

void RawWindow::paintGL()
{
    // Window translation but none seems to be performed
    glViewport(0, 0, width(), height());

    // Bind framebuffer to the context, "not necessary to call this function in
    // most cases, because it is called automatically before invoking
    // paintGL()."
    makeCurrent();

    // Clear buffer
    glClear(GL_COLOR_BUFFER_BIT);

    // Binds the vertex array object to the OpenGL binding point
    Vao.bind();
    Program->bind();

    // Map resources for CUDA
    cudaSafeCall(cudaGraphicsMapResources(1, &cuResource, cuStream));
    // Retrive the cuda pointer
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer(&cuPtrToPbo, &sizeBuffer, cuResource));

    // Get the last image from the ouput queue
    void* frame = output_->get_last_image();

    // Put the frame inside the cuda ressrouce

    if (api::get_img_type() == ImgType::Composite)
        cudaXMemcpyAsync(cuPtrToPbo, frame, sizeBuffer, cudaMemcpyDeviceToDevice, cuStream);
    else
        convert_frame_for_display(frame, cuPtrToPbo, fd_.get_frame_res(), fd_.depth, 0, cuStream);

    // Release resources (needs to be done at each call) and sync
    cudaSafeCall(cudaGraphicsUnmapResources(1, &cuResource, cuStream));
    cudaXStreamSynchronize(cuStream);

    // Texture creationg
    glBindTexture(GL_TEXTURE_2D, Tex);

    // Binds buffer to texture data source
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, Pbo);

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fd_.width, fd_.height, texType, texDepth, nullptr);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

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

void RawWindow::mousePressEvent(QMouseEvent* e) { overlay_manager_.press(e); }

void RawWindow::mouseMoveEvent(QMouseEvent* e) { overlay_manager_.move(e); }

void RawWindow::mouseReleaseEvent(QMouseEvent* e)
{
    if (e->button() == Qt::LeftButton)
        overlay_manager_.release(fd_.width);
    else if (e->button() == Qt::RightButton)
        resetTransform();
}

void RawWindow::keyPressEvent(QKeyEvent* e)
{
    BasicOpenGLWindow::keyPressEvent(e);

    float translation = translation_step_ / scale_;

    switch (e->key())
    {
    case Qt::Key::Key_8:
        translate_[1] -= translation;
        break;
    case Qt::Key::Key_2:
        translate_[1] += translation;
        break;
    case Qt::Key::Key_6:
        translate_[0] += translation;
        break;
    case Qt::Key::Key_4:
        translate_[0] -= translation;
        break;
    }
    setTransform();
}

void RawWindow::zoomInRect(units::RectOpengl zone)
{
    const units::PointOpengl center = zone.center();

    const float delta_x = center.x() / (getScale() * 2);
    const float delta_y = center.y() / (getScale() * 2);

    const auto old_translate = getTranslate();

    const auto new_translate_x = old_translate[0] + delta_x;
    const auto new_translate_y = old_translate[1] - delta_y;

    setTranslate(new_translate_x, new_translate_y);

    const float xRatio = zone.unsigned_width();

    // Use the commented line below if you are using square windows,
    // and comment the 2 lines below
    const float yRatio = zone.unsigned_height();
    setScale(getScale() / (std::min(xRatio, yRatio) / 2));

    setTransform();
}

bool RawWindow::is_resize_call() const { return is_resize; }

void RawWindow::set_is_resize(bool b) { is_resize = b; }

void RawWindow::wheelEvent(QWheelEvent* e)
{
    QPointF pos = e->position();
    if (!is_between(static_cast<int>(pos.x()), 0, width()) || !is_between(static_cast<int>(pos.y()), 0, height()))
        return;
    const float xGL = (static_cast<float>(pos.x() - width() / 2)) / static_cast<float>(width()) * 2.f;
    const float yGL = -((static_cast<float>(pos.y() - height() / 2)) / static_cast<float>(height())) * 2.f;
    if (e->angleDelta().y() > 0)
    {
        scale_ += 0.1f * scale_;
        translate_[0] += xGL * 0.1 / scale_;
        translate_[1] += -yGL * 0.1 / scale_;
        setTransform();
    }
    else if (e->angleDelta().y() < 0)
    {
        scale_ -= 0.1f * scale_;
        if (scale_ < 1.f)
            scale_ = 1;
        else
        {
            translate_[0] -= xGL * 0.1 / scale_;
            translate_[1] -= -yGL * 0.1 / scale_;
            setTransform();
        }
    }
}

void RawWindow::closeEvent(QCloseEvent* event)
{
    if (kView == KindOfView::Raw || kView == KindOfView::Hologram)
    {
        save_gui("holo window");
    }
}

void RawWindow::save_gui(std::string window)
{
    // Don't forget to test the cases where the window is out ouf the screen boundaries
    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    j_us[window]["width"] = width();
    j_us[window]["height"] = height();
    j_us[window]["x"] = x();
    j_us[window]["y"] = y();

    std::ofstream output_file(path);
    output_file << j_us.dump(1);
}
} // namespace holovibes::gui
