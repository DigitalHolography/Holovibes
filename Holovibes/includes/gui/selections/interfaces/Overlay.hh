/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Interface for all overlays.*/
#pragma once

#include <array>

#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>

#include "frame_desc.hh"
#include "rect.hh"
#include "compute_descriptor.hh"

namespace holovibes
{
namespace gui
{

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
    // Composite overlays
    CompositeArea,
    Rainbow
};

class BasicOpenGLWindow;

using Color = std::array<float, 3>;

class Overlay : protected QOpenGLFunctions
{
  public:
    Overlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);
    virtual ~Overlay();

    /*! \brief Get the zone selected

    */
    const units::RectFd& getZone() const;

    /*! \brief Get the kind of overlay

    */
    const KindOfOverlay getKind() const;

    /*! \brief Return if the overlay should be displayed

    */
    const bool isDisplayed() const;
    /*! \brief Return if the overlay have to be deleted

    */
    const bool isActive() const;
    /*! \brief Disable this overlay

    */
    void disable();
    /*! \brief Enable this overlay

    */
    void enable();

    /*! \brief Initialize shaders and Vao/Vbo of the overlay

    */
    void initProgram();

    /*! \brief Call opengl function to draw the overlay

    */
    virtual void draw() = 0;

    /*! \brief Called when the user press the mouse button

    */
    virtual void press(QMouseEvent* e);
    /*! \brief Called when the user press a key

    */
    virtual void keyPress(QKeyEvent* e);
    /*! \brief Called when the user moves the mouse

    */
    virtual void move(QMouseEvent* e) = 0;
    /*! \brief Called when the user release the mouse button

    */
    virtual void release(ushort frameside) = 0;

    /*! \brief Prints informations about the overlay. Debug purpose.

    */
    void print();

  protected:
    /*! \brief Initialize Vao/Vbo

    */
    virtual void init() = 0;

    /*! \brief Convert the current zone into opengl coordinates (-1, 1) and set
       the vertex buffer

    */
    virtual void setBuffer() = 0;

    /*! \brief Converts QPoint to a point in the window

    */
    units::PointWindow getMousePos(const QPoint& pos);

    //! Zone selected by the users in pixel coordinates (window width, window
    //! height)
    units::RectFd zone_;

    //! Kind of overlay
    KindOfOverlay kOverlay_;

    //! Indexes of the buffers in opengl
    GLuint verticesIndex_, colorIndex_, elemIndex_;
    //! Specific Vao of the overlay
    QOpenGLVertexArrayObject Vao_;
    //! The opengl shader program
    std::unique_ptr<QOpenGLShaderProgram> Program_;
    //! Location of the vertices buffer in the shader/vertexattrib. Set to 2
    unsigned short verticesShader_;
    //! Location of the color buffer in the shader/vertexattrib. Set to 3
    unsigned short colorShader_;

    //! The color of the overlay. Each component must be between 0 and 1.
    Color color_;
    //! Transparency of the overlay, between 0 and 1
    float alpha_;

    /*! If the overlay is activated or not.
     *  Since we don't want the overlay to remove itself from the vector of
     * overlays, We set this boolean, and remove it later by iterating through
     * the vector.
     */
    bool active_;
    //! If the overlay should be displayed or not
    bool display_;
    //! Pointer to the parent to access Compute descriptor and Pipe
    BasicOpenGLWindow* parent_;
};
} // namespace gui
} // namespace holovibes
