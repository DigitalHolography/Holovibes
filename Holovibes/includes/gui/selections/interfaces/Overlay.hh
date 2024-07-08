/*! \file
 *
 * \brief Interface for all overlays in the Holovibes GUI.
 */
#pragma once

#include <array>

#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>

#include "frame_desc.hh"
#include "rect.hh"

namespace holovibes::gui
{

/*! \enum KindOfOverlay
 *
 * \brief Enumerates the different types of overlays.
 */
enum KindOfOverlay
{
    Zoom,
    Reticle,
    // Chart
    Signal,
    Noise,
    // Cross
    Cross,
    SliceCross,
    // Filter2D
    Filter2DReticle,
    // Composite overlays
    CompositeArea,
    Rainbow
};

class BasicOpenGLWindow;

using Color = std::array<float, 3>;

/*! \class Overlay
 *
 * \brief Base class for all overlays in the Holovibes GUI.
 *
 * This class provides the interface and common functionality for different types of overlays,
 * including managing OpenGL resources, handling user input, and rendering.
 */
class Overlay : protected QOpenGLFunctions
{
  public:
    /*! \brief Constructor
     *
     * \param overlay The kind of overlay.
     * \param parent Pointer to the parent BasicOpenGLWindow.
     */
    Overlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);

    /*! \brief Destructor */
    virtual ~Overlay();

    /*! \brief Gets the selected zone.
     *
     * \return The selected zone in frame descriptor coordinates.
     */
    const units::RectFd& getZone() const;

    /*! \brief Gets the kind of overlay.
     *
     * \return The kind of overlay.
     */
    const KindOfOverlay getKind() const;

    /*! \brief Checks if the overlay should be displayed.
     *
     * \return True if the overlay should be displayed, false otherwise.
     */
    const bool isDisplayed() const;

    /*! \brief Checks if the overlay is active and should not be deleted.
     *
     * \return True if the overlay is active, false otherwise.
     */
    const bool isActive() const;

    /*! \brief Disables the overlay. */
    void disable();

    /*! \brief Enables the overlay. */
    void enable();

    /*! \brief Called when the overlay is set as current. */
    virtual void onSetCurrent();

    /*! \brief Initializes the shaders and VAO/VBO for the overlay. */
    void initProgram();

    /*! \brief Draws the overlay using OpenGL functions. */
    virtual void draw() = 0;

    /*! \brief Handles mouse press events.
     *
     * \param e Pointer to the QMouseEvent.
     */
    virtual void press(QMouseEvent* e);

    /*! \brief Handles key press events.
     *
     * \param e Pointer to the QKeyEvent.
     */
    virtual void keyPress(QKeyEvent* e);

    /*! \brief Handles mouse move events.
     *
     * \param e Pointer to the QMouseEvent.
     */
    virtual void move(QMouseEvent* e) = 0;

    /*! \brief Handles mouse release events.
     *
     * \param frameside The side of the frame where the release occurred.
     */
    virtual void release(ushort frameside) = 0;

    /*! \brief Prints information about the overlay for debugging purposes. */
    void print();

  protected:
    /*! \brief Initializes the VAO/VBO. */
    virtual void init() = 0;

    /*! \brief Converts the current zone into OpenGL coordinates and sets the vertex buffer. */
    virtual void setBuffer() = 0;

    /*! \brief Converts a QPoint to a point in the window.
     *
     * \param pos The QPoint to convert.
     * \return The corresponding point in the window coordinates.
     */
    units::PointWindow getMousePos(const QPoint& pos);

    units::RectFd zone_; /*!< Selected zone in pixel coordinates (window width, window height). */
    KindOfOverlay kOverlay_; /*!< Kind of overlay. */

    GLuint verticesIndex_, colorIndex_, elemIndex_; /*!< OpenGL buffer indices. */
    QOpenGLVertexArrayObject Vao_; /*!< OpenGL Vertex Array Object. */
    std::unique_ptr<QOpenGLShaderProgram> Program_; /*!< OpenGL shader program. */
    unsigned short verticesShader_; /*!< Shader location for vertices. */
    unsigned short colorShader_; /*!< Shader location for color. */

    Color color_; /*!< Color of the overlay. Each component must be between 0 and 1. */
    float alpha_; /*!< Transparency of the overlay, between 0 and 1. */

    bool active_; /*!< Indicates if the overlay is active and should not be deleted. */
    bool display_; /*!< Indicates if the overlay should be displayed. */
    BasicOpenGLWindow* parent_; /*!< Pointer to the parent window. */
};
} // namespace holovibes::gui