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

namespace holovibes
{
/*! \brief Contains all function to display the graphical user interface */
namespace gui
{
/*! \enum KindOfView
 *
 * \brief Describes the kind of window
 */
enum class KindOfView
{
    Raw = 1,  /*!< Simply displaying the input frames */
    Hologram, /*!< Applying the demodulation and computations on the input frames */
    Lens,     /*!< Displaying the FFT1/FFT2 lens view */
    SliceXZ,  /*!< Displaying the XZ view of the hologram */
    SliceYZ,  /*!< Displaying the YZ view of the hologram */
    Filter2D, /*!< Displaying the Filter2D view of the hologram */
};

/*! \class BasicOpenGLWindow
 *
 * \brief #TODO Add a description for this class
 */
class BasicOpenGLWindow : public QOpenGLWindow, protected QOpenGLFunctions
{
  public:
    // Constructor & Destructor
    BasicOpenGLWindow(QPoint p, QSize s, ComputeDescriptor* cd, DisplayQueue* q, KindOfView k);
    virtual ~BasicOpenGLWindow();

    const KindOfView getKindOfView() const;
    const KindOfOverlay getKindOfOverlay() const;
    void resetSelection();

    ComputeDescriptor* getCd();
    const ComputeDescriptor* getCd() const;
    const camera::FrameDescriptor& getFd() const;
    OverlayManager& getOverlayManager();

    /*! \name Transform functions
     * \{
     */
    virtual void resetTransform();
    void setScale(float);
    float getScale() const;
    void setAngle(float a);
    float getAngle() const;
    void setFlip(bool f);
    bool getFlip() const;
    void setTranslate(float x, float y);
    glm::vec2 getTranslate() const;

    const glm::mat3x3& getTransformMatrix() const;
    const glm::mat3x3& getTransformInverseMatrix() const;
    /*! \} */

  protected:
    Qt::WindowState winState;
    QPoint winPos;

    /*! \brief Output queue filled in the computing pipeline */
    DisplayQueue* output_;
    ComputeDescriptor* cd_;
    const camera::FrameDescriptor& fd_;
    const KindOfView kView;

    OverlayManager overlay_manager_;

    virtual void setTransform();

    /*! \name CUDA Objects
     * \{
     */
    cudaGraphicsResource_t cuResource;
    cudaStream_t cuStream;
    /*! \} */

    void* cuPtrToPbo;
    size_t sizeBuffer;

    /*! \name OpenGL Objects
     * \{
     */
    QOpenGLShaderProgram* Program;
    QOpenGLVertexArrayObject Vao;
    GLuint Vbo, Ebo, Pbo;
    GLuint Tex;
    /*! \} */

    /*! \name Virtual Pure Functions
     * \{
     */
    virtual void initShaders() = 0;
    virtual void initializeGL() = 0;
    virtual void resizeGL(int width, int height);
    virtual void paintGL() = 0;
    /*! \} */

    /*! \name Event functions
     * \{
     */
    void timerEvent(QTimerEvent* e);
    virtual void keyPressEvent(QKeyEvent* e);
    virtual bool eventFilter(QObject* obj, QEvent* event) override;
    /*! \} */

  protected:
    glm::vec4 translate_;
    float scale_;
    /*! \brief Angle in degree */
    float angle_;
    bool flip_;

    glm::mat3x3 transform_matrix_;
    glm::mat3x3 transform_inverse_matrix_;
};
} // namespace gui
} // namespace holovibes
