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

/*! \file
 *
 * Contains compute parameters. */
#pragma once

# include <atomic>
# include <mutex>
# include <tuple>

# include "observable.hh"
# include "Rectangle.hh"

namespace holovibes
{
	const static std::string version = "v4.3.170426"; /*!< Current version of this project. */
	#ifndef TUPLE4F
	# define TUPLE4F
		using	Tuple4f =	std::tuple<float, float, float, float>;
	#endif
	using	CameraKind =
	enum
	{
		NONE,
		Adimec,
		Edge,
		IDS,
		Ixon,
		Pike,
		Pixelfly,
		xiQ
	};
	using	Algorithm =
	enum
	{
		None,
		FFT1,
		FFT2
	};
	using	Computation =
	enum
	{
		Stop,
		Direct,
		Hologram
	};
	using	ComplexViewMode =
	enum
	{
		Modulus,
		SquaredModulus,
		Argument,
		PhaseIncrease,
		Complex
	};
	using	AccessMode =
	enum
	{
		Get = 1,
		Set
	};
	using	WindowKind =
	enum
	{
		MainDisplay,
		SliceXZ,
		SliceYZ
	};
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

		QPoint stft_slice_cursor;
		/*! Average mode signal zone */
		gui::Rectangle signal_zone;
		/*! Average mode noise zone */
		gui::Rectangle noise_zone;
		/*! Selected zone in which apply the autofocus algorithm. */
		gui::Rectangle autofocus_zone;
		/*! Selected zone in which apply the stft algorithm. */
		gui::Rectangle stft_roi_zone;

	public:
		/*! \brief ComputeDescriptor constructor
		 * Initialize the compute descriptor to default values of computation. */
		ComputeDescriptor();

		/*! \brief Assignment operator
		 * The assignment operator is explicitely defined because std::atomic type
		 * does not allow to generate assignments operator automatically. */
		ComputeDescriptor& operator=(const ComputeDescriptor& cd);
		
		void reset();

		void stftCursor(QPoint *p, AccessMode m);

		void signalZone(gui::Rectangle& rect, AccessMode m);
		
		void noiseZone(gui::Rectangle& rect, AccessMode m);

		void autofocusZone(gui::Rectangle& rect, AccessMode m);

		void stftRoiZone(gui::Rectangle& rect, AccessMode m);

		#pragma region Atomics vars
		/*! Hologram algorithm. */
		std::atomic<Algorithm> algorithm;
		/*! Computing mode used by the pipe */
		std::atomic<Computation> compute_mode;
		/*! Complex to float method. */
		std::atomic<ComplexViewMode> view_mode;

		std::atomic<WindowKind> current_window;

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
		/*! Contrast minimal and maximal range value. */
		std::atomic<float> contrast_min;
		std::atomic<float> contrast_max;
		std::atomic<float> contrast_min_slice_xz;
		std::atomic<float> contrast_min_slice_yz;
		std::atomic<float> contrast_max_slice_xz;
		std::atomic<float> contrast_max_slice_yz;
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
		std::atomic<unsigned int> img_acc_cutsXZ_level;
		std::atomic<unsigned int> img_acc_cutsYZ_level;
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
		/*! History of unwrap size */
		std::atomic<int> unwrap_history_size;
		/*! Special buffer size*/
		std::atomic<int> special_buffer_size;

		/*! Is convolution processing enabled. */
		std::atomic<bool> convolution_enabled;
		/*! Is convolution processing enabled. */
		std::atomic<bool> flowgraphy_enabled;
		/*! Is log scale post-processing enabled. */
		std::atomic<bool> log_scale_enabled;
		std::atomic<bool> log_scale_enabled_cut_xz;
		std::atomic<bool> log_scale_enabled_cut_yz;
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
		std::atomic<bool> img_acc_cutsXZ_enabled;
		std::atomic<bool> img_acc_cutsYZ_enabled;
		std::atomic<ushort> p_accu_enabled;
		std::atomic<ushort> p_accu_min_level;
		std::atomic<ushort> p_accu_max_level;
		#pragma endregion
	};
}
