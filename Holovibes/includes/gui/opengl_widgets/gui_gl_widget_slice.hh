#pragma once

# include <array>
# include <QGLWidget>
# include <QShortcut.h>
# include <QOpenGLFunctions.h>
# include <QTimer>
# include <QMouseEvent>

# include <cuda_gl_interop.h>

# include "frame_desc.hh"
# include "geometry.hh"
# include "holovibes.hh"

/* Forward declaration. */
namespace holovibes
{
	class Queue;
}

namespace gui
{
	/*! \brief OpenGL widget used to display frames contained in Queue(s).
	 *
	 * Users can select zone and move in display surf
	 * Selected zone with mouse will emit qt signals.
	 */
	class GLWidgetSlice : public QGLWidget, protected QOpenGLFunctions
	{
		Q_OBJECT

		/*! Frame rate of the display in Hertz (Frame.s-1) */
		const unsigned int DISPLAY_FRAMERATE = 30;

	#pragma region Public Methods
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
		GLWidgetSlice(
			holovibes::Holovibes& h,
			holovibes::Queue& q,
			const unsigned int width,
			const unsigned int height,
			QWidget* parent = 0);

		virtual ~GLWidgetSlice();

		/*! \brief This property holds the recommended minimum size for the widget. */
		QSize minimumSizeHint() const override;

		/*! \brief This property holds the recommended size for the widget. */
		QSize sizeHint() const override;

		/*! \brief enable selection mode */
		void enable_selection()
		{
			is_selection_enabled_ = true;
		}

		const gui::Rectangle& get_signal_selection() const
		{
			return signal_selection_;
		}

		const gui::Rectangle& get_noise_selection() const
		{
			return noise_selection_;
		}

		void set_signal_selection(const gui::Rectangle& selection)
		{
			signal_selection_ = selection;
			//h_.get_compute_desc().signal_zone = signal_selection_;
			h_.get_compute_desc().signalZone(&signal_selection_, holovibes::ComputeDescriptor::Set);
		}

		void set_noise_selection(const gui::Rectangle& selection)
		{
			noise_selection_ = selection;
			//h_.get_compute_desc().noise_zone = noise_selection_;
			h_.get_compute_desc().signalZone(&noise_selection_, holovibes::ComputeDescriptor::Set);

		}
	#pragma endregion
	/* ---------- */
	#pragma region Slots & Signals
	public slots:
		void resizeFromWindow(const int width, const int height);

		/*! \{ \name View Shortcut */
		void view_move_down();
		void view_move_left();
		void view_move_right();
		void view_move_up();
		void view_zoom_in();
		void view_zoom_out();
		/*! \} */

	signals:

		/*! \brief Signal used to inform the main window that roi
		** zone is definitely selected.
		*/
		void stft_roi_zone_selected_end();
	#pragma endregion
	/* ---------- */
	#pragma region Protected Methods
	protected:
		/* \brief Initialize all OpenGL components needed */
		void initializeGL() override;

		/*! \brief Called whenever the OpenGL widget is resized */
		void resizeGL(int width, int height) override;

		/*! \brief Paint the scene and the selection zone(s) according to selection_mode_.
		**
		** The image is painted directly from the GPU, avoiding several
		** back and forths memory transfers.
		** This method uses the NVI idiom with set_texture_format by wrapping it
		** with common boilerplate code.
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
	#pragma endregion
	/* ---------- */
	#pragma region Protected Fields
	protected:
		QWidget* parent_;
		holovibes::Holovibes& h_;
		holovibes::Queue&     queue_;
		//!< Metadata on the images received for display.
		const camera::FrameDescriptor&  frame_desc_;

		/*! \brief QTimer used to refresh the OpenGL widget */
		QTimer timer_;

		/*! \{ \name OpenGl graphique buffer */
		GLuint  buffer_;
		struct cudaGraphicsResource*  cuda_buffer_;
		cudaStream_t cuda_stream_; //!< Drawing operates on a individual stream.
		/*! \} */

		/*! \{ \name Selection */
		/*! \brief User is currently select zone ? */
		bool is_selection_enabled_;
		/*! \brief Boolean used to switch between signal and noise selection */
		bool is_signal_selection_;
		/*! \} */

		/*! \{ \name Selection */
		/*! \brief Current selection */
		gui::Rectangle selection_;
		gui::Rectangle signal_selection_;
		gui::Rectangle noise_selection_;
		gui::Rectangle stft_roi_selection_;
		/*! \} */

		/*! \{ \name Previouses zoom translations */
		float px_;
		float py_;
		float zoom_ratio_;
		/*! \} */

		/*! \{ \name Window size hints */
		const unsigned int width_;
		const unsigned int height_;
		/*! \} */

		/*! \{ \name Key shortcut */
		QShortcut *num_2_shortcut;
		QShortcut *num_4_shortcut;
		QShortcut *num_6_shortcut;
		QShortcut *num_8_shortcut;
		QShortcut *key_plus_shortcut;
		QShortcut *key_minus_shortcut;
		/*! \} */
	#pragma endregion
	/* ---------- */
	#pragma region Private Methods
	private:
		/*! \brief Draw a selection zone
		**
		** Coordinates are first converted to OpenGL ones then previous translations and scales
		** due to zooms are respectively canceled in order for the zone to be at the user's coordinates.
		**
		** \param selection zone to draw
		** \param color color of the zone to draw in [red, green, blue, alpha] format
		*/
		void selection_rect(const gui::Rectangle& selection, const float color[4]);

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
		void zoom(const gui::Rectangle& selection);

		/*! \brief Dezoom to default resolution */
		void dezoom();

		/*! \brief Return resized rectangle using actual zoom */
//		gui::Rectangle  GLWidget::resize_zone(gui::Rectangle selection);

		/*! \brief Assure that the rectangle starts at topLeft and ends at bottomRight
		** no matter what direction the user uses to select a zone.
		*/
		void swap_selection_corners(gui::Rectangle& selection);

		/*! \brief Ensure that selection zone is in widget's bounds i-e camera's resolution */
		void bounds_check(gui::Rectangle& selection);

		/*! \brief Check glError and print then
		 *
		 * Use only in debug mode, glGetError is slow and should be avoided
		 */
		void gl_error_checking();
	#pragma endregion
	};
}