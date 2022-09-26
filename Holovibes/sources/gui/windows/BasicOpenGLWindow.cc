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

namespace holovibes
{
using camera::FrameDescriptor;
namespace gui
{
BasicOpenGLWindow::BasicOpenGLWindow(QPoint p, QSize s, DisplayQueue* q, KindOfView k)
    : QOpenGLWindow()
    , QOpenGLFunctions()
    , winState(Qt::WindowNoState)
    , winPos(p)
    , output_(q)
    , fd_(q->get_fd())
    , kView(k)
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
    , translate_(0.f, 0.f, 0.f, 0.f)
    , scale_(1.f)
    , angle_(0.f)
    , flip_(0)
    , transform_matrix_(1.0f)
    , transform_inverse_matrix_(1.0f)
{
    cudaSafeCall(cudaStreamCreateWithPriority(&cuStream, cudaStreamDefault, CUDA_STREAM_WINDOW_PRIORITY));
    resize(s);
    setFramePosition(p);
    setIcon(QIcon(":/holovibes_logo.png"));
    this->installEventFilter(this);
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

const KindOfView BasicOpenGLWindow::getKindOfView() const { return kView; }

const KindOfOverlay BasicOpenGLWindow::getKindOfOverlay() const { return overlay_manager_.getKind(); }

const FrameDescriptor& BasicOpenGLWindow::getFd() const { return fd_; }

OverlayManager& BasicOpenGLWindow::getOverlayManager() { return overlay_manager_; }

const glm::mat3x3& BasicOpenGLWindow::getTransformMatrix() const { return transform_matrix_; }

const glm::mat3x3& BasicOpenGLWindow::getTransformInverseMatrix() const { return transform_inverse_matrix_; }

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

void BasicOpenGLWindow::setAngle(float a)
{
    angle_ = a;
    setTransform();
}

float BasicOpenGLWindow::getAngle() const { return angle_; }

void BasicOpenGLWindow::setFlip(bool f)
{
    flip_ = f;
    setTransform();
}

bool BasicOpenGLWindow::getFlip() const { return flip_; }

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
    flip_ = false;
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
    LOG_FUNC(main, angle_, flip_);

    const glm::mat4 rotY = glm::rotate(glm::mat4(1.f), glm::radians(180.f * (flip_ == 1)), glm::vec3(0.f, 1.f, 0.f));
    const glm::mat4 rotZ = glm::rotate(glm::mat4(1.f), glm::radians(angle_), glm::vec3(0.f, 0.f, 1.f));
    glm::mat4 rotYZ = rotY * rotZ;

    // Avoid float multiplication imprecision due to glm::rotate
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            rotYZ[i][j] = std::round(rotYZ[i][j]);

    const glm::mat4 scl = glm::scale(
        glm::mat4(1.f),
        glm::vec3(kView == KindOfView::SliceYZ ? 1 : scale_, kView == KindOfView::SliceXZ ? 1 : scale_, 1.f));
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
        Program->setUniformValue(Program->uniformLocation("angle"), angle_);
        Program->setUniformValue(Program->uniformLocation("flip"), flip_);
        Program->setUniformValue(Program->uniformLocation("translate"), trs[0], trs[1]);
        QMatrix4x4 m(glm::value_ptr(mvp));
        Program->setUniformValue(Program->uniformLocation("mvp"), m.transposed());
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
} // namespace gui
} // namespace holovibes
