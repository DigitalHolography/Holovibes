#ifdef WIN32
#include <windows.h>
#endif
#include <cuda_gl_interop.h>

#include <QGuiApplication>
#include <QMouseEvent>
#include <QPointF>
#include <QRect>
#include <QScreen>
#include <QWheelEvent>

#include "RawWindow.hh"
#include "HoloWindow.hh"
#include "cuda_memory.cuh"
#include "texture_update.cuh"
#include "common.cuh"
#include "tools.hh"
#include "API.hh"
#include "logger.hh"

#include "GUI.hh"
#include "user_interface_descriptor.hh"
#include "notifier.hh"
#include "rect_gl.hh"

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
        gui::create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / "vertex.raw.glsl").string()));
    Program->addShaderFromSourceFile(
        QOpenGLShader::Fragment,
        gui::create_absolute_qt_path(RELATIVE_PATH(__SHADER_FOLDER_PATH__ / "fragment.tex.raw.glsl").string()));
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
    cudaTexture = new CudaTexture(fd_.width, fd_.height, fd_.depth, cuStream);
    if (!cudaTexture->init())
    {
        LOG_ERROR("Failed to initialize CUDA Texture");
    }

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
    startTimer(1000 / UserInterfaceDescriptor::instance().display_rate_);
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

    if ((API.compute.get_compute_mode() == Computation::Hologram &&
         API.transform.get_space_transformation() == SpaceTransformation::NONE) ||
        API.compute.get_compute_mode() == Computation::Raw)
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
    void* frame = output_->get_last_image();
    if (!frame)
        return;

    glViewport(0, 0, width(), height());
    makeCurrent();
    glClear(GL_COLOR_BUFFER_BIT);

    Vao.bind();
    Program->bind();

    cudaTexture->update(frame, fd_);

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

void RawWindow::zoomInRect(units::RectFd zone)
{
    const RectGL converted(*this, zone);

    const PointGL center = converted.center();

    const float delta_x = center.x() / (getScale() * 2);
    const float delta_y = center.y() / (getScale() * 2);

    const auto old_translate = getTranslate();

    const auto new_translate_x = old_translate[0] + delta_x;
    const auto new_translate_y = old_translate[1] - delta_y;

    setTranslate(new_translate_x, new_translate_y);

    const float xRatio = converted.unsigned_width();

    // Use the commented line below if you are using square windows,
    // and comment the 2 lines below
    const float yRatio = converted.unsigned_height();
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

    // If raw view closed, deactivate it and update the ui
    if (kView == KindOfView::Raw)
    {
        API.view.set_raw_view(false);
        gui::set_raw_view(false, 0);
        NotifierManager::notify("notify", true);
    }

    // If lens view closed, deactivate it and update the ui
    else if (kView == KindOfView::Lens)
    {
        API.view.set_lens_view(false);
        gui::set_lens_view(false, 0);
        NotifierManager::notify("notify", true);
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
