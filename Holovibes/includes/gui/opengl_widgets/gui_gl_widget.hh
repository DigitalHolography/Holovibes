/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#pragma once

# include <array>
# include <QGLWidget>
# include <QShortcut.h>
# include <QOpenGLFunctions.h>
# include <QTimer>
# include <QMouseEvent>

# include <cuda_gl_interop.h>

# include "frame_desc.hh"
# include "holovibes.hh"

/* Forward declaration. */
//namespace holovibes
//{
//	class Queue;
//}

namespace gui
{
	/*! Zone selection modes */
	typedef enum selection
	{
		/*AUTOFOCUS,
		AVERAGE,
		ZOOM,
		STFT_ROI,
		STFT_SLICE*/
		gn
	} eselection;

	/*! \brief OpenGL widget used to display frames contained in Queue(s).
	 *
	 * Users can select zone and move in display surf
	 * Selected zone with mouse will emit qt signals.
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
		** \param h holovibes object
		** \param q Queue containing the frames to display
		** \param width widget's width
		** \param height widget's height
		** \param parent Qt parent (should be a GUIGlWindow)
		*/
		GLWidget(
			holovibes::Holovibes& h,
			holovibes::Queue& q,
			const unsigned int width,
			const unsigned int height,
			QWidget* parent = 0);

		virtual ~GLWidget();

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
			h_.get_compute_desc().signalZone(&signal_selection_, holovibes::ComputeDescriptor::Set);
		}

		void set_noise_selection(const gui::Rectangle& selection)
		{
			noise_selection_ = selection;
			h_.get_compute_desc().signalZone(&noise_selection_, holovibes::ComputeDescriptor::Set);
		}

		void set_selection_mode(const eselection mode)
		{
			selection_mode_ = mode;
		}

		//			getKindOfSelection
		/*eselection	get_selection_mode(void)
		{
			holovibes::ComputeDescriptor& cd = h_.get_compute_desc();
			if (cd.stft_view_enabled.load())
				return (eselection::STFT_SLICE);
			else if (cd.filter_2d_enabled.load())
				return (eselection::STFT_ROI);
			else if (cd.average_enabled.load())
				return (eselection::AVERAGE);
			else
				return (eselection::ZOOM);

			return selection_mode_;
		}*/

		float px_;
		float py_;
		/*! \brief Dezoom to default resolution */
		void dezoom();

		QString	windowTitle;
	public slots:
		void resizeFromWindow(const int width, const int height);

		/*! \{ \name View Shortcut */
		void view_move_down();
		void view_move_left();
		void view_move_right();
		void view_move_up();
		void view_zoom_in();
		void view_zoom_out();
		void block_slice();
		/*! \} */


	signals:
		/*! \brief Signal used to inform the main window that autofocus
		** zone has been selected.
		*/
		void autofocus_zone_selected(gui::Rectangle zone);

		/*! \brief Signal used to inform the main window that roi
		** zone has been selected but not definitely.
		*/
		void stft_roi_zone_selected_update(gui::Rectangle zone);

		/*! \brief Signal used to inform the main window that roi
		** zone is definitely selected.
		*/
		void stft_roi_zone_selected_end();

		/*TODO:*/
		void stft_slice_pos_update(QPoint pos);

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
		/*! \brief Color zone and signal emit depend of this */
		eselection selection_mode_;
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
		QShortcut *key_space_shortcut;
		/*! \} */

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
		

		/*! \brief Return resized rectangle using actual zoom */
		gui::Rectangle  GLWidget::resize_zone(gui::Rectangle selection);

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

		bool		slice_block_;
	};
}