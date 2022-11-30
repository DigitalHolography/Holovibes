#include <map>

#include <QGuiApplication>
#include <QKeyEvent>
#include <QRect>
#include <QScreen>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "texture_update.cuh"
#include "BasicOpenGLWindow.hh"
#include "HoloWindow.hh"

#include "holovibes.hh"
#include "tools.hh"
#include "API.hh"

namespace holovibes::api
{
static WindowKind get_window_kind_form_kind_of_window(gui::KindOfView kview)
{
    static auto map = std::map<gui::KindOfView, WindowKind>{{gui::KindOfView::Raw, WindowKind::ViewXY},
                                                            {gui::KindOfView::Hologram, WindowKind::ViewXY},
                                                            {gui::KindOfView::SliceXZ, WindowKind::ViewXZ},
                                                            {gui::KindOfView::SliceYZ, WindowKind::ViewYZ},
                                                            {gui::KindOfView::ViewFilter2D, WindowKind::ViewFilter2D}};
    if (map.contains(kview) == false)
        throw std::runtime_error("Expect WindowKind view");
    return map[kview];
}

static const ViewXYZ& get_view_as_xyz_type(gui::KindOfView kind)
{
    return api::get_view_as_xyz_type(get_window_kind_form_kind_of_window(kind));
}
} // namespace holovibes::api

namespace holovibes::gui
{
BasicOpenGLWindow::BasicOpenGLWindow(QPoint p, QSize s, DisplayQueue* q, KindOfView k)
    : QOpenGLWindow()
    , QOpenGLFunctions()
    , winState(Qt::WindowNoState)
    , winPos(p)
    , winSize(s)
    , output_(q)
    , fd_(q->get_fd())
    , overlay_manager_(this)
    , cuResource(nullptr)
    , cuPtrToPbo(nullptr)
    , sizeBuffer(0)
    , Program(nullptr)
    , Vao(0)
    , Vbo(0)
    , Ebo(0)
    , Pbo(0)
    , Tex(0)
    , transform_matrix_(1.0f)
    , transform_inverse_matrix_(1.0f)
    , translate_(0.f, 0.f, 0.f, 0.f)
    , scale_(1.f)
    , kind_of_view(k)
{
    LOG_FUNC();
    cudaSafeCall(cudaStreamCreateWithPriority(&cuStream, cudaStreamDefault, CUDA_STREAM_WINDOW_PRIORITY));

    resize(winSize);
    setFramePosition(winPos);
    setIcon(QIcon(":/holovibes_logo.png"));
    installEventFilter(this);
}

BasicOpenGLWindow::~BasicOpenGLWindow()
{
    makeCurrent();

    cudaStreamDestroy(cuStream);

    if (Tex)
        glDeleteBuffers(1, &Tex);
    if (Pbo)
        glDeleteBuffers(1, &Pbo);
    if (Ebo)
        glDeleteBuffers(1, &Ebo);
    if (Vbo)
        glDeleteBuffers(1, &Vbo);
    Vao.destroy();
    delete Program;
}

void BasicOpenGLWindow::resizeGL(int width, int height)
{
    if (winState == Qt::WindowFullScreen)
        return;
    glViewport(0, 0, width, height);
}

void BasicOpenGLWindow::timerEvent(QTimerEvent* e) { QPaintDeviceWindow::update(); }

void BasicOpenGLWindow::keyPressEvent(QKeyEvent* e)
{
    switch (e->key())
    {
    case Qt::Key::Key_F11:
        winState = winState == Qt::WindowFullScreen ? Qt::WindowNoState : Qt::WindowFullScreen;
        setWindowState(winState);
        break;
    case Qt::Key::Key_Escape:
        winPos = QPoint(0, 0);
        winState = Qt::WindowNoState;
        setWindowState(winState);
        break;
    }
    overlay_manager_.keyPress(e);
}

void BasicOpenGLWindow::setTranslate(float x, float y)
{
    translate_[0] = x;
    translate_[1] = y;
    setTransform();
}

glm::vec2 BasicOpenGLWindow::getTranslate() const { return glm::vec2(translate_[0], translate_[1]); }

void BasicOpenGLWindow::resetTransform()
{
    translate_ = {0.f, 0.f, 0.f, 0.f};
    scale_ = 1.f;
    setTransform();
}

void BasicOpenGLWindow::setScale(float scale)
{
    scale_ = scale;
    setTransform();
}

float BasicOpenGLWindow::getScale() const { return scale_; }

void BasicOpenGLWindow::setTransform()
{
    LOG_FUNC();

    // FIXME API-FIXME VIEW : View should be the same
    glm::mat4 rotY;
    glm::mat4 rotZ;
    if (kind_of_view == KindOfView::SliceYZ)
    {
        rotY = glm::rotate(glm::mat4(1.f),
                           glm::radians(180.f * (api::get_view_as_xyz_type(kind_of_view).horizontal_flip ? 1 : 0)),
                           glm::vec3(0.f, 1.f, 0.f));

        rotZ = glm::rotate(glm::mat4(1.f),
                           glm::radians(api::get_view_as_xyz_type(kind_of_view).rotation),
                           glm::vec3(0.f, 0.f, 1.f));
    }
    else if (kind_of_view != KindOfView::Lens)
    {
        rotY = glm::rotate(glm::mat4(1.f),
                           glm::radians(180.f * (api::get_view_as_xyz_type(kind_of_view).horizontal_flip ? 0 : 1)),
                           glm::vec3(0.f, 1.f, 0.f));
        rotZ = glm::rotate(glm::mat4(1.f),
                           glm::radians(api::get_view_as_xyz_type(kind_of_view).rotation),
                           glm::vec3(0.f, 0.f, 1.f));
    }
    else
    {
        rotY = glm::rotate(glm::mat4(1.f), glm::radians(0.f), glm::vec3(0.f, 1.f, 0.f));
        rotZ = glm::rotate(glm::mat4(1.f), glm::radians(0.f), glm::vec3(0.f, 0.f, 1.f));
    }

    glm::mat4 rotYZ = rotY * rotZ;

    // Avoid float multiplication imprecision due to glm::rotate
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            rotYZ[i][j] = std::round(rotYZ[i][j]);

    const glm::mat4 scl = glm::scale(glm::mat4(1.f),
                                     glm::vec3(kind_of_view == KindOfView::SliceYZ ? 1 : scale_,
                                               kind_of_view == KindOfView::SliceXZ ? 1 : scale_,
                                               1.f));
    glm::mat4 mvp = rotYZ * scl;

    for (uint id = 0; id < 2; id++)
        if (is_between(translate_[id], -FLT_EPSILON, FLT_EPSILON))
            translate_[id] = 0.f;

    glm::vec4 trs = rotYZ * translate_;
    transform_matrix_ = mvp;
    // GLM matrix are column major so the translation vector is in [2][X] and
    // not [X][2]
    transform_matrix_[2][0] = -translate_[0] * 2 * scale_;
    transform_matrix_[2][1] = translate_[1] * 2 * scale_;

    transform_matrix_[2][2] = 1;

    transform_inverse_matrix_ = glm::inverse(transform_matrix_);
    if (Program)
    {
        makeCurrent();
        Program->bind();
        QMatrix4x4 m(glm::value_ptr(mvp));
        Program->setUniformValue(Program->uniformLocation("mvp"), m.transposed());
        Program->setUniformValue(Program->uniformLocation("translate"), trs[0], trs[1]);

        if (kind_of_view == KindOfView::Raw)
        {
            Program->setUniformValue(Program->uniformLocation("bitshift"), api::get_raw_bitshift());
            return;
        }
        else
            Program->setUniformValue(Program->uniformLocation("bitshift"), 0);

        Program->release();
    }
}

void BasicOpenGLWindow::resetSelection() { overlay_manager_.reset(); }

bool BasicOpenGLWindow::eventFilter(QObject* obj, QEvent* event)
{
    if (event->type() == QEvent::Close)
    {
        emit destroyed();
    }

    return QObject::eventFilter(obj, event);
}
} // namespace holovibes::gui
