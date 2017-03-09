/*! \file
 *
 * Contains compute parameters. */
#pragma once

# include <atomic>
# include <mutex>
# include <QPoint>

# include "observable.hh"
# include "geometry.hh"

using guard = std::lock_guard<std::mutex>;

namespace holovibes
{
	const static std::string version = "v4.2.170308"; /*!< Current version of this project. */


	/*! \brief Contains compute parameters.
	 *
	 * Theses parameters will be used when the pipe is refresh.
	 * It defines parameters for FFT, lens (Fresnel transforms ...),
	 * post-processing (contrast, shift_corners, log scale).
	 *
	 * The class use the *Observer* design pattern instead of the signal
	 * mechanism of Qt because classes in the namespace holovibes are
	 * independent of GUI or CLI implementations. So that, the code remains
	 * reusable.
	 *
	 * This class contains std::atomic fields to avoid concurrent access between
	 * the pipe and the GUI.
	 */
	class ComputeDescriptor : public Observable
	{
	private:
		mutable std::mutex mutex_;

		/* TODO:*/
		//std::atomic<QPoint> stft_slice_cursor;
		QPoint stft_slice_cursor;
		/*! Average mode signal zone */
		//std::atomic<Rectangle> signal_zone;
		Rectangle signal_zone;
		/*! Selected zone in which apply the autofocus algorithm. */
		//std::atomic<Rectangle> autofocus_zone;
		Rectangle autofocus_zone;
		/*! Average mode noise zone */
		//std::atomic<Rectangle> noise_zone;
		Rectangle noise_zone;
		/*! Selected zone in which apply the stft algorithm. */
		//std::atomic<Rectangle> stft_roi_zone;
		Rectangle stft_roi_zone;

	public:
		#pragma region enums
		/*! \brief Select hologram methods. */
		enum fft_algorithm
		{
			None,
			FFT1,
			FFT2
		};

		/*! \brief select which mode the pipe will be using*/
		enum compute_mode
		{
			DIRECT,
			HOLOGRAM
		};

		/*! \brief Complex to float methods.
		 *
		 * Select the method to apply to transform a complex hologram frame to a
		 * float frame. */
		enum complex_view_mode
		{
			MODULUS,
			SQUARED_MODULUS,
			ARGUMENT,
			COMPLEX,
			PHASE_INCREASE,
		};

		typedef
		enum	e_access
		{
			Get = 1,
			Set = 2
		}		t_access;
		#pragma endregion

		/*! \brief ComputeDescriptor constructor
		 * Initialize the compute descriptor to default values of computation. */
		ComputeDescriptor();

		/*! \brief Assignment operator
		 * The assignment operator is explicitely defined because std::atomic type
		 * does not allow to generate assignments operator automatically. */
		ComputeDescriptor& operator=(const ComputeDescriptor& cd);
		
		void stftCursor(QPoint *p, t_access mode);

		void signalZone(Rectangle *rect, t_access mode);
		
		void noiseZone(Rectangle *rect, t_access mode);

		void autofocusZone(Rectangle *rect, t_access mode);

		void stftRoiZone(Rectangle *rect, t_access mode);

		#pragma region Atomics vars
		/*! Hologram algorithm. */
		std::atomic<enum fft_algorithm> algorithm;
		/*! Computing mode used by the pipe */
		std::atomic<enum compute_mode> compute_mode;
		/*! Complex to float method. */
		std::atomic<enum complex_view_mode> view_mode;

		/*! Number of samples in which apply the fft on. */
		std::atomic<unsigned short> nsamples;
		/*! p-th output component to show. */
		std::atomic<unsigned short> pindex;
		/*! q-th output component of FFT to use with vibrometry. */
		std::atomic<unsigned short> vibrometry_q;

		/*! Lambda in meters. */
		std::atomic<float> lambda;
		/*! Computing mode used by the pipe */
		std::vector<float> convo_matrix;
		/*! Sensor-to-object distance. */
		std::atomic<float> zdistance;
		/*! Contrast minimal range value. */
		std::atomic<float> contrast_min;
		/*! Contrast maximal range value. */
		std::atomic<float> contrast_max;
		/*! Z minimal range for autofocus. */
		std::atomic<float> autofocus_z_min;
		/*! Z maximal range for autofocus. */
		std::atomic<float> autofocus_z_max;
		/*! Pixel Size used when importing a file */
		std::atomic<float> import_pixel_size;

		/*! Size of Image Accumulation buffer. */
		std::atomic<unsigned int> img_acc_buffer_size;
		/*! Convolution matrix length. */
		std::atomic<unsigned int> convo_matrix_width;
		/*! Convolution matrix height. */
		std::atomic<unsigned int> convo_matrix_height;
		/*! Convolution matrix z. */
		std::atomic<unsigned int> convo_matrix_z;
		/*! Set Flowgraphy level: */
		std::atomic<unsigned int> flowgraphy_level;
		/*! Set Image Accumulation level. */
		std::atomic<unsigned int> img_acc_level;
		/*! Height of the matrix used inside the autofocus calculus. */
		std::atomic<unsigned int> autofocus_size;
		/*! Number of points of autofocus between the z range. */
		std::atomic<unsigned int> autofocus_z_div;
		/*! Number of iterations of autofocus between the z range. */
		std::atomic<unsigned int> autofocus_z_iter;

		/*! Quantity of frames that will be processed during STFT. */
		std::atomic<int> stft_level;
		/*! Quantity frames to wait in STFT mode before computing a temporal FFT. */
		std::atomic<int> stft_steps;
		/*! Frame number of images that will be averaged. */
		std::atomic<int> ref_diff_level;
		/*! TODO: */
		std::atomic<int> unwrap_history_size;
		/*! Special buffer size*/
		std::atomic<int> special_buffer_size;

		/*! Is convolution processing enabled. */
		std::atomic<bool> convolution_enabled;
		/*! Is convolution processing enabled. */
		std::atomic<bool> flowgraphy_enabled;
		/*! Is log scale post-processing enabled. */
		std::atomic<bool> log_scale_enabled;
		/*! Is FFT shift corners post-processing enabled. */
		std::atomic<bool> shift_corners_enabled;
		/*! Is manual contrast post-processing enabled. */
		std::atomic<bool> contrast_enabled;
		/*! Is stft mode enabled. */
		std::atomic<bool> stft_enabled;
		/*! Is vibrometry method enabled. */
		std::atomic<bool> vibrometry_enabled;
		/*! Is reference difference mode enabled. */
		std::atomic<bool> ref_diff_enabled;
		/* Is reference  slinding difference mode enabled */
		std::atomic<bool> ref_sliding_enabled;
		/*! Is filter2D enabled. */
		std::atomic<bool> filter_2d_enabled;
		/*! Is stft view enabled. */
		std::atomic<bool> stft_view_enabled;
		/*! Is average enabled. */
		std::atomic<bool> average_enabled;
		/*! Is signal trig enabled. */
		std::atomic<bool> signal_trig_enabled;
		/*! Is read file a .cine file. */
		std::atomic<bool> is_cine_file;
		/*! Is Image Accumulation enabled. */
		std::atomic<bool> img_acc_enabled;
		#pragma endregion
	};
}