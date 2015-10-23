#ifndef GUI_GL_WIDGET_HH_
# define GUI_GL_WIDGET_HH_

# include <array>
# include <QGLWidget>
# include <QShortcut.h>
# include <QOpenGLFunctions.h>
# include <QTimer>
# include <QMouseEvent>

# include <cuda_gl_interop.h>

# include "queue.hh"
# include <frame_desc.hh>
# include "geometry.hh"
# include "holovibes.hh"

namespace gui
{
  /*! Zone selection modes */
  typedef enum selection
  {
    AUTOFOCUS,
    AVERAGE,
    ZOOM,
    STFT_ROI,
  } eselection;

  /*! \class GLWidget
  **
  ** OpenGL widget used to display frames contained in Queue(s).
  */
  class GLWidget : public QGLWidget, protected QOpenGLFunctions
  {
    Q_OBJECT

    /*! Frame rate of the display in Hertz (Frame.s-1) */
    const unsigned int DISPLAY_FRAMERATE = 30;

  public:
    /* \brief GLWidget constructor
    **
    ** Build the widget and start a display QTimer.
    **
    ** \param h holovibes object
    ** \param q Queue containing the frames to display
    ** \param width widget's width
    ** \param height widget's height
    ** \param parent Qt parent (should be a GUIGlWindow)
    */
    GLWidget(
      holovibes::Holovibes& h,
      holovibes::Queue& q,
      unsigned int width,
      unsigned int height,
      QWidget* parent = 0);

    ~GLWidget();
    QSize minimumSizeHint() const;
    QSize sizeHint() const;

    /*! \brief enable selection mode */
    void enable_selection()
    {
      is_selection_enabled_ = true;
    }

    const holovibes::Rectangle& get_signal_selection() const
    {
      return signal_selection_;
    }

    const holovibes::Rectangle& get_noise_selection() const
    {
      return noise_selection_;
    }

    void set_signal_selection(const holovibes::Rectangle& selection)
    {
      signal_selection_ = selection;
      h_.get_compute_desc().signal_zone = signal_selection_;
    }

    void set_noise_selection(const holovibes::Rectangle& selection)
    {
      noise_selection_ = selection;
      h_.get_compute_desc().noise_zone = noise_selection_;
    }

    void set_selection_mode(eselection mode)
    {
      selection_mode_ = mode;
    }

    public slots:
    void resizeFromWindow(int width, int height);

    /*! \{ \name View Shortcut */
    void view_move_down();
    void view_move_left();
    void view_move_right();
    void view_move_up();
    void view_zoom_in();
    void view_zoom_out();
    /*! \} */
  signals:
    /*! \brief Signal used to inform the main window that autofocus
    ** zone has been selected.
    */
    void autofocus_zone_selected(holovibes::Rectangle zone);
    void stft_roi_zone_selected_update(holovibes::Rectangle zone);
    void stft_roi_zone_selected_end();

  protected:
    /* \brief Initialize all OpenGL components needed */
    void initializeGL() override;
    /*! \brief Called whenever the OpenGL widget is resized */
    void resizeGL(int width, int height) override;
    /*! \brief Paint the scene and the selection zone(s) according to selection_mode_
    **
    ** Scene is painted directly from GPU. It avoid several back and forths memory transfers.
    */
    void paintGL() override;

    /*! \brief Starts selection
    **
    ** Whenever mouse is pressed, the selection rectangle top left corner is
    ** defined at the current coordinates. If the zoom mode is active and right
    ** mous button is pressed then dezoom occured.
    */
    void mousePressEvent(QMouseEvent* e) override;
    /*! \brief Change selection rectangle bottom right corner */
    void mouseMoveEvent(QMouseEvent* e) override;
    /*! \brief Ends selection
    **
    ** Whenever mouse is released, selection bottom right corner is set to current
    ** mouse coordinates then a bound check is done then a swap of corners if necessary.
    **
    ** In AUTOFOCUS mode, a signal is sent to the main window to inform that selection is
    ** done.
    ** In AVERAGE mode, it is alternatively signal and zone selection that are set.
    ** In ZOOM mode, its check that the selection is not a point.
    */
    void mouseReleaseEvent(QMouseEvent* e) override;

  private:
    /*! \brief Draw a selection zone
    **
    ** Coordinates are first converted to OpenGL ones then previous translations and scales
    ** due to zooms are respectively canceled in order for the zone to be at the user's coordinates.
    **
    ** \param selection zone to draw
    ** \param color color of the zone to draw in [red, green, blue, alpha] format
    */
    void selection_rect(const holovibes::Rectangle& selection, float color[4]);
    /*! \brief Zoom to a given zone
    **
    ** Selection coordinates are first converted to OpenGL ones.
    ** Then the center of the selection zone has to move to the center of the GLWidget,
    ** a classic (xb - xa, yb - ya) calculus gives the translation vector.
    ** Then a zoom ratio is computed using the camera's resolution and the selection's
    ** dimensions.
    ** Then the frame is scaled to the previous ratio.
    **
    ** All the translations and scales (zoom ratios) are stored in order for the next selection
    ** zones to be displayed correctly.
    **
    ** \param selection zone where to zoom
    */
    void zoom(const holovibes::Rectangle& selection);
    /*! \brief Dezoom to default resolution */
    void dezoom();

    /*! \brief Return resized rectangle using actual zoom */
    holovibes::Rectangle  GLWidget::resize_zone(holovibes::Rectangle selection);
    /*! \brief Assure that the rectangle starts at topLeft and ends at bottomRight
    ** no matter what direction the user uses to select a zone.
    */
    void swap_selection_corners(holovibes::Rectangle& selection);

    /*! \brief Ensure that selection zone is in widget's bounds i-e camera's resolution */
    void bounds_check(holovibes::Rectangle& selection);

    void gl_error_checking();

  private:
    holovibes::Holovibes& h_;

    /*! QTimer used to refresh the OpenGL widget */
    QTimer timer_;
    bool is_selection_enabled_;
    holovibes::Rectangle selection_;
    eselection selection_mode_;
    /*! Boolean used to switch between signal and noise selection */
    bool is_signal_selection_;
    holovibes::Rectangle signal_selection_;
    holovibes::Rectangle noise_selection_;
    holovibes::Rectangle stft_roi_selection_;
    /*! Base view */
    holovibes::Rectangle base_view_;
    QWidget* parent_;

    /*! /{ \name Previouses zoom translations */
    float px_;
    float py_;
    /*! \} */

    /*! Previouses zoom ratios */
    float zoom_ratio_;

    /*! \{ \name View Shortcut */
    QShortcut	*num_2_shortcut;
    QShortcut	*num_4_shortcut;
    QShortcut	*num_6_shortcut;
    QShortcut	*num_8_shortcut;
    QShortcut	*zoom_in_shortcut;
    QShortcut	*zoom_out_shortcut;
    /*! \} */

    /*! \{ \name Window size hints */
    unsigned int width_;
    unsigned int height_;
    /*! \} */

    holovibes::Queue& queue_;
    const camera::FrameDescriptor& frame_desc_;
    GLuint buffer_;
    struct cudaGraphicsResource* cuda_buffer_;
  };
}

#endif /* !GUI_GL_WIDGET_HH_ */