/*! \file
 *
 * \brief Interface for all overlays.
 *
 * You can change the position and scale of the overlay either:
 * - By changing the `zone_` variable which hold a src point and a dst point.
 * - By changing the `translation_` and the `scale_` variable. These are in OpenGL clip space. So the scale should be in
 * [0, 1] and the translation in [-1, 1]. By default the overlay is at the center of the screen with a scale of 1.
 *
 * In the end the `zone_` will be converted to `translation_` and `scale_`.
 */
#pragma once

#include <array>

#include <glm/glm.hpp>

#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QDateTime>

#include "frame_desc.hh"
#include "rect.hh"

namespace holovibes::gui
{

/*! \enum KindOfOverlay
 *
 * \brief Holds all the kind of overlay. It's then used to create overlay with the overlay_manager
 */
enum KindOfOverlay
{
    Zoom,
    Reticle,
    // Registration
    Registration,
    // Chart
    Signal,
    Noise,
    // Cross
    Cross,
    SliceCross,
    // Composite overlays
    CompositeArea,
    Rainbow
};

class BasicOpenGLWindow;

using Color = std::array<float, 3>;

/*! \class Overlay
 *
 * \brief class that represents an overlay in the window.
 */
class Overlay : protected QOpenGLFunctions
{
  public:
    Overlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);
    virtual ~Overlay();

    /*! \brief Get the kind of overlay */
    const inline KindOfOverlay getKind() const { return kOverlay_; }

    /*! \brief Return if the overlay should be displayed */
    const inline bool isDisplayed() const { return display_; }

    /*! \brief Return if the overlay have to be deleted */
    const inline bool isActive() const { return active_; }

    /*! \brief Get the time before the overlay will hide */
    const inline QDateTime getTimeBeforeHide() const { return time_before_hide_; }

    /*! \brief Set the time before the overlay will hide */
    void inline setTimeBeforeHide(QDateTime time) { time_before_hide_ = time; }

    /*! \brief Disable this overlay */
    void disable();
    /*! \brief Enable this overlay */
    void enable();

    /*! \brief Called when the overlay is set as current */
    virtual void onSetCurrent();

    /*! \brief Initialize shaders and Vao/Vbo of the overlay */
    void initProgram();

    /*! \brief Call opengl function to draw the overlay */
    virtual void draw() = 0;

    /*! \brief Called when the user press the mouse button */
    virtual void press(QMouseEvent* e);
    /*! \brief Called when the user press a key */
    virtual void keyPress(QKeyEvent* e);
    /*! \brief Called when the user moves the mouse */
    virtual void move(QMouseEvent* e) = 0;
    /*! \brief Called when the user release the mouse button */
    virtual void release(ushort frameside) = 0;

    /*! \brief Prints informations about the overlay. Debug purpose. */
    void print();

  protected:
    /*! \brief Initialize Vao/Vbo */
    virtual void init() = 0;

    /*! \brief Convert the current zone into opengl coordinates (-1, 1) and set the vertex buffer */
    virtual void setBuffer() = 0;

    /*! \brief Setup context, shaders for drawing and bind ressources */
    void initDraw();

    /*! \brief Unbind ressources */
    void endDraw();

    /*! \brief Converts QPoint to a point in the window */
    units::PointWindow getMousePos(const QPoint& pos);

    /*! \brief Zone selected by the users in pixel coordinates (window width, window height) */
    units::RectFd zone_;

    /*! \brief Kind of overlay */
    KindOfOverlay kOverlay_;

    /*! \brief Indexes of the buffers in opengl */
    GLuint verticesIndex_, colorIndex_, elemIndex_;
    /*! \brief Specific Vao of the overlay */
    QOpenGLVertexArrayObject Vao_;
    /*! \brief The opengl shader program */
    std::unique_ptr<QOpenGLShaderProgram> Program_;
    /*! \brief Location of the vertices buffer in the shader/vertexattrib. Set to 2 */
    unsigned short verticesShader_;
    /*! \brief Location of the color buffer in the shader/vertexattrib. Set to 3 */
    unsigned short colorShader_;

    /*! \brief The color of the overlay. Each component must be between 0 and 1. */
    Color color_;
    /*! \brief Transparency of the overlay, between 0 and 1 */
    float alpha_;

    /*! \brief The time in ms when the overlay will disappear */
    QDateTime time_before_hide_;

    /*! \brief If the overlay is activated or not.
     *
     * Since we don't want the overlay to remove itself from the vector of
     * overlays, We set this boolean, and remove it later by iterating through
     * the vector.
     */
    bool active_;
    /*! \brief If the overlay should be displayed or not */
    bool display_;
    /*! \brief Pointer to the parent to access Compute descriptor and Pipe */
    BasicOpenGLWindow* parent_;

    /*! \brief The scale of the overlays */
    glm::vec2 scale_;

    /*! \brief The translation of the overlays */
    glm::vec2 translation_;
};
} // namespace holovibes::gui
