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
	const static std::string version = "v4.3.170612"; /*!< Current version of this project. */

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
		mutable std::mutex	mutex_;

		QPoint				stft_slice_cursor;

		gui::Rectangle		signal_zone;
		gui::Rectangle		noise_zone;
		gui::Rectangle		autofocus_zone;
		gui::Rectangle		stft_roi_zone;

	public:
		/*! \brief ComputeDescriptor constructor
		 * Initialize the compute descriptor to default values of computation. */
		ComputeDescriptor();
		~ComputeDescriptor();

		/*! \brief Assignment operator
		 * The assignment operator is explicitely defined because std::atomic type
		 * does not allow to generate assignments operator automatically. */
		ComputeDescriptor& operator=(const ComputeDescriptor& cd);

		void stftCursor(QPoint *p, AccessMode m);
		void signalZone(gui::Rectangle& rect, AccessMode m);
		void noiseZone(gui::Rectangle& rect, AccessMode m);
		void autofocusZone(gui::Rectangle& rect, AccessMode m);
		void stftRoiZone(gui::Rectangle& rect, AccessMode m);

		#pragma region Atomics vars
		std::atomic<Algorithm>			algorithm;
		std::atomic<Computation>		compute_mode;
		std::atomic<ComplexViewMode>	view_mode;
		std::atomic<bool>				vision_3d_enabled;
		std::atomic<WindowKind>			current_window;
		std::atomic<ushort>				nsamples;
		std::atomic<ushort>				pindex;
		std::atomic<ushort>				vibrometry_q;
		std::atomic<float>				lambda;
		std::vector<float>				convo_matrix;
		std::atomic<float>				zdistance;
		std::atomic<float>				contrast_min;
		std::atomic<float>				contrast_max;
		std::atomic<float>				contrast_min_slice_xz;
		std::atomic<float>				contrast_min_slice_yz;
		std::atomic<float>				contrast_max_slice_xz;
		std::atomic<float>				contrast_max_slice_yz;
		std::atomic<ushort>				cuts_contrast_p_offset;
		std::atomic<float>				autofocus_z_min;
		std::atomic<float>				autofocus_z_max;
		std::atomic<float>				import_pixel_size;
		std::atomic<uint>				img_acc_buffer_size;
		std::atomic<uint>				convo_matrix_width;
		std::atomic<uint>				convo_matrix_height;
		std::atomic<uint>				convo_matrix_z;
		std::atomic<uint>				flowgraphy_level;
		std::atomic<uint>				img_acc_level;
		std::atomic<uint>				img_acc_cutsXZ_level;
		std::atomic<uint>				img_acc_cutsYZ_level;
		std::atomic<uint>				autofocus_size;
		std::atomic<uint>				autofocus_z_div;
		std::atomic<uint>				autofocus_z_iter;
		std::atomic<int>				stft_level;
		std::atomic<int>				stft_steps;
		std::atomic<int>				ref_diff_level;
		std::atomic<int>				unwrap_history_size;
		std::atomic<int>				special_buffer_size;
		std::atomic<bool>				convolution_enabled;
		std::atomic<bool>				flowgraphy_enabled;
		std::atomic<bool>				log_scale_enabled;
		std::atomic<bool>				log_scale_enabled_cut_xz;
		std::atomic<bool>				log_scale_enabled_cut_yz;
		std::atomic<bool>				shift_corners_enabled;
		std::atomic<bool>				contrast_enabled;
		std::atomic<bool>				stft_enabled;
		std::atomic<bool>				vibrometry_enabled;
		std::atomic<bool>				ref_diff_enabled;
		std::atomic<bool>				ref_sliding_enabled;
		std::atomic<bool>				filter_2d_enabled;
		std::atomic<bool>				stft_view_enabled;
		std::atomic<bool>				average_enabled;
		std::atomic<bool>				signal_trig_enabled;
		std::atomic<bool>				is_cine_file;
		std::atomic<bool>				img_acc_enabled;
		std::atomic<bool>				img_acc_cutsXZ_enabled;
		std::atomic<bool>				img_acc_cutsYZ_enabled;
		std::atomic<ushort>				p_accu_enabled;
		std::atomic<ushort>				p_accu_min_level;
		std::atomic<ushort>				p_accu_max_level;
		std::atomic<float>				display_rate;

		#pragma endregion
	};
}
