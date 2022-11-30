/*! \file
 *
 * \brief Interface implemented by each Qt window.
 */
#pragma once

#include <glm/glm.hpp>

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWindow>

#include "overlay_manager.hh"
#include "tools_conversion.cuh"
#include "display_queue.hh"
#include "enum_window_kind.hh"

/*! \brief Contains all function to display the graphical user interface */
namespace holovibes::gui
{
/*! \enum KindOfView
 *
 * \brief Describes the kind of window
 */
enum class KindOfView
{
    Raw = 1,      /*!< Simply displaying the input frames */
    Hologram,     /*!< Applying the demodulation and ComputeModeEnums on the input frames */
    Lens,         /*!< Displaying the FFT1/FFT2 lens view */
    SliceXZ,      /*!< Displaying the XZ view of the hologram */
    SliceYZ,      /*!< Displaying the YZ view of the hologram */
    ViewFilter2D, /*!< Displaying the ViewFilter2D view of the hologram */
};

/*! \class BasicOpenGLWindow
 *
 * \brief #TODO Add a description for this class
 */
class BasicOpenGLWindow : public QOpenGLWindow, protected QOpenGLFunctions
{
  public:
  public:
    // Constructor & Destructor
    BasicOpenGLWindow(QPoint p, QSize s, DisplayQueue* q, KindOfView k);
    virtual ~BasicOpenGLWindow();

    const KindOfView getKindOfView() const { return kind_of_view; }
    const KindOfOverlay getKindOfOverlay() const { return overlay_manager_.getKind(); }
    const FrameDescriptor& getFd() const { return fd_; }
    OverlayManager& getOverlayManager() { return overlay_manager_; }

    const QPoint& get_window_position() const { return winPos; }
    QPoint& get_window_position() { return winPos; }
    const QSize& get_window_size() const { return winSize; }
    QSize& get_window_size() { return winSize; }

    void resetSelection();

  public:
    /*! \name Transform functions
     * \{
     */
    void resetTransform();
    void setScale(float);
    float getScale() const;
    void setTranslate(float x, float y);
    glm::vec2 getTranslate() const;

    const glm::mat3x3& getTransformMatrix() const { return transform_matrix_; }
    const glm::mat3x3& getTransformInverseMatrix() const { return transform_inverse_matrix_; }
    /*! \} */

    virtual void setTransform();

  protected:
    /*! \name Virtual Pure Functions (trick used because Qt define these functions, please implement them)
     * \{
     */
    virtual void initShaders() = 0;
    void initializeGL() override {}
    void paintGL() override {}
    /*! \} */

    void resizeGL(int width, int height) override;

    /*! \name Event functions
     * \{
     */
    void timerEvent(QTimerEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;
    bool eventFilter(QObject* obj, QEvent* event) override;
    /*! \} */

  protected:
    Qt::WindowState winState;
    QPoint winPos;
    QSize winSize;

    /*! \brief Output queue filled in the computing pipeline */
    DisplayQueue* output_;
    const FrameDescriptor& fd_;

    OverlayManager overlay_manager_;
    /*! \name CUDA Objects
     * \{
     */
    cudaGraphicsResource_t cuResource;
    void* cuPtrToPbo;
    cudaStream_t cuStream;
    size_t sizeBuffer;
    /*! \} */

    /*! \name OpenGL Objects
     * \{
     */
    QOpenGLShaderProgram* Program;
    QOpenGLVertexArrayObject Vao;
    GLuint Vbo, Ebo, Pbo;
    GLuint Tex;
    /*! \} */

  protected:
    glm::mat3x3 transform_matrix_;
    glm::mat3x3 transform_inverse_matrix_;

    glm::vec4 translate_;
    float scale_;
    const KindOfView kind_of_view;
    bool has_been_load_ = false;
};
} // namespace holovibes::gui
