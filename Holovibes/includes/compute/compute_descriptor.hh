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

# include "observable.hh"
# include "rect.hh"

namespace holovibes
{
	const static std::string version = "v5.8"; /*!< Current version of this project. */

	using	Tuple4f =	std::tuple<float, float, float, float>;

	/*!
	 * \brief	Difference kind of camera supported by Holovibes
	 */
	enum CameraKind
	{
		NONE,
		Adimec,
		Edge,
		IDS,
		Ixon,
		Hamamatsu,
		Pike,
		Pixelfly,
		xiQ,
		xiB,
		PhotonFocus
	};

	/*! \brief	Rendering mode for Hologram */
	enum Algorithm
	{
		None, /**< Nothing Applied */
		FFT1, /**< Fresnel Transform */
		FFT2  /**< Angular spectrum propagation */
	};

	/*! \bried	Input processes */
	enum Computation
	{
		Stop, /**< Input not displayed */
		Direct, /**< Interferogram recorded */
		Hologram /**< Reconstruction of the object */
	};

	/*! \brief	Displaying type of the image */
	enum ImgType
	{
		Modulus, /**< Modulus of the complex data */
		SquaredModulus, /**< Modulus taken to its square value */
		Argument, /**< Phase (angle) value of the complex pixel c, computed with atan(Im(c)/Re(c)) */
		PhaseIncrease, /**< Phase value computed with the conjugate between the phase of the last image and the previous one */
		Complex, /**< Displays the complex buffer using blue and red colors for real and imaginary part */
		Composite /**< Displays different frequency intervals on color chanels*/
	};

	/*! \brief Describes the access mode of an accessor. */
	enum AccessMode
	{
		Get = 1,
		Set
	};

	/*!
	 * \brief	Represents the kind of slice displayed by the window
	 */
	enum WindowKind
	{
		XYview,
		XZview,
		YZview
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
		/*! \brief The lock used in the zone accessors */
		mutable std::mutex	mutex_;


		/*! \brief	The position of the point used to obtain XZ and YZ views */
		units::PointFd		stft_slice_cursor;

		/*! \brief	The zone to average the signal */
		units::RectFd		signal_zone;
		/*! \brief	The zone to average the noise */
		units::RectFd		noise_zone;
		/*! \brief	The zone used to compute automatically the z-value */
		units::RectFd		autofocus_zone;
		/*! \brief	Limits the computation to only this zone. Also called Filter 2D*/
		units::RectFd		stft_roi_zone;
		/*! \brief	The area on which we'll normalize the colors*/
		units::RectFd		composite_zone;
		/*! \brief	The area on which we'll run the convolution to stabilize*/
		units::RectFd		stabilization_zone;
		/*! \brief  The area used to limit the stft computations. */
		units::RectFd		zoomed_zone;

	public:
		/*! \brief ComputeDescriptor constructor
		 * Initialize the compute descriptor to default values of computation. */
		ComputeDescriptor();

		/*! \brief ComputeDescriptor destructor.
		
		*/
		~ComputeDescriptor();

		/*! \brief Assignment operator
		 * The assignment operator is explicitely defined because std::atomic type
		 * does not allow to generate assignments operator automatically. */
		ComputeDescriptor& operator=(const ComputeDescriptor& cd);

		/*!
		 * @{
		 *
		 * \brief	Accessor to the selected zone
		 *
		 * \param			rect	The rectangle to process
		 * \param 		  	m   	An AccessMode to process.
		 */

		void signalZone(units::RectFd& rect, AccessMode m);
		void noiseZone(units::RectFd& rect, AccessMode m);
		void autofocusZone(units::RectFd& rect, AccessMode m);
		//! @}

		/*!
		 * @{
		 *
		 * \brief	Getter of the overlay positions.
		 *
		 */

		units::RectFd getStftZone() const;
		units::RectFd getCompositeZone() const;
		units::RectFd getStabilizationZone() const;
		units::RectFd getZoomedZone() const;
		units::PointFd getStftCursor() const;
		//! @}

		/*!
		 * @{
		 *
		 * \brief	Setter of the overlay positions.
		 *
		 */
		void setStftZone(const units::RectFd& rect);
		void setCompositeZone(const units::RectFd& rect);
		void setStabilizationZone(const units::RectFd& rect);
		void setZoomedZone(const units::RectFd& rect);
		void setStftCursor(const units::PointFd& rect);
		//! @}

		#pragma region Atomics vars
		//! Algorithm to apply in hologram mode
		std::atomic<Algorithm>		algorithm;
		//! Mode of computation of the image
		std::atomic<Computation>	compute_mode;
		//! type of the image displayed
		std::atomic<ImgType>		img_type;

		//! Last window selected
		std::atomic<WindowKind>		current_window;
		//! Number of images used by SFTF i.e. depth of the SFTF cube
		std::atomic<ushort>			nSize;
		//! index in the depth axis
		std::atomic<ushort>			pindex;
		std::atomic<ushort>			vibrometry_q;
		//! wave length of the laser
		std::atomic<float>			lambda;
		//! Input matrix used for convolution
		std::vector<float>			convo_matrix;
		//! z value used by fresnel transform
		std::atomic<float>			zdistance;

		//! minimum constrast value in xy view
		std::atomic<float>			contrast_min_slice_xy;
		//! maximum constrast value in xy view
		std::atomic<float>			contrast_max_slice_xy;
		//! minimum constrast value in xz view
		std::atomic<float>			contrast_min_slice_xz;
		//! maximum constrast value in xz view
		std::atomic<float>			contrast_max_slice_xz;
		//! minimum constrast value in yz view
		std::atomic<float>			contrast_min_slice_yz;
		//! maximum constrast value in yz view
		std::atomic<float>			contrast_max_slice_yz;

		//! minimum autofocus value in xy view
		std::atomic<float>			autofocus_z_min;
		//! maximum constrast value in xy view
		std::atomic<float>			autofocus_z_max;

		std::atomic<ushort>			cuts_contrast_p_offset;
		//! Size of a pixel in micron
		std::atomic<float>			pixel_size;
		//! Correction factor of the scale bar, used to match the objective of the camera
		std::atomic<float>			scale_bar_correction_factor;
		//! Width of the matrix used for convolution
		std::atomic<uint>			convo_matrix_width;
		//! Height of the matric used for convolution
		std::atomic<uint>			convo_matrix_height;
		std::atomic<uint>			convo_matrix_z;
		std::atomic<uint>			flowgraphy_level;
		std::atomic<uint>			autofocus_size;
		/*! Number of divison of zmax - zmin used by the autofocus algorithm */
		std::atomic<uint>			autofocus_z_div;
		/*! Number of loops done by the autofocus algorithm */
		std::atomic<uint>			autofocus_z_iter;
		//! Size of the stft_queue.
		std::atomic<int>			stft_level;
		//! Number of pipe iterations between two temporal demodulation.
		std::atomic<int>			stft_steps;
		std::atomic<int>			unwrap_history_size;
		std::atomic<int>			special_buffer_size;
		//! is convolution enabled
		std::atomic<bool>			convolution_enabled;
		//! is flowgraphy enabled
		std::atomic<bool>			flowgraphy_enabled;
		//! is vibrometry enabled
		std::atomic<bool>			vibrometry_enabled;
		//! is log scale in slice XY enabled
		std::atomic<bool>			log_scale_slice_xy_enabled;
		//! is log scale in slice XZ enabled
		std::atomic<bool>			log_scale_slice_xz_enabled;
		//! is log scale in slice YZ enabled
		std::atomic<bool>			log_scale_slice_yz_enabled;
		//! is shift fft enabled (switching representation diagram) 
		std::atomic<bool>			shift_corners_enabled;
		//! enables the contract for the slice xy, yz and xz
		std::atomic<bool>			contrast_enabled;
		//! enable the limitation of the stft to the zoomed area.
		std::atomic<bool>			croped_stft;
		//! Enables the difference with the selected frame.
		std::atomic<bool>			ref_diff_enabled;
		//! Enabled the difference with the ref_diff_level previous frame
		std::atomic<bool>			ref_sliding_enabled;
		std::atomic<int>			ref_diff_level;
		//! allows to limit the computations to a selected zone
		std::atomic<bool>			filter_2d_enabled;
		//! are slices YZ and XZ enabled
		std::atomic<bool>			stft_view_enabled;
		//! is gpu lens display activated
		std::atomic<bool>			gpu_lens_display_enabled { true };
		//! enables the signal and noise average computation
		std::atomic<bool>			average_enabled;

		//! is file a .cine
		std::atomic<bool>			is_cine_file;

		//! Number of frame per seconds displayed
		std::atomic<float>			display_rate;

		//! Enables the XY stabilization.
		std::atomic<bool>			xy_stabilization_enabled;
		//! Pause the stabilization, in order to select the stabilization area
		std::atomic<bool>			xy_stabilization_paused;
		//! Displays the convolution matrix.
		std::atomic<bool>			xy_stabilization_show_convolution;

		//! Enables the interpolation, to match the real pixel size according to the laser wavelength.
		std::atomic<bool>			interpolation_enabled;
		//! Current wavelength of the laser
		std::atomic<float>			interp_lambda;
		//! Initial wavelength of the laser
		std::atomic<float>			interp_lambda1;
		//! Final wavelength of the laser
		std::atomic<float>			interp_lambda2;
		std::atomic<float>			interp_sensitivity;
		std::atomic<int>			interp_shift;

		//! is img average in view XY enabled (average of output over time, i.e. phase compensation)
		std::atomic<bool>			img_acc_slice_xy_enabled;
		//! is img average in view XZ enabled
		std::atomic<bool>			img_acc_slice_xz_enabled;
		//! is img average in view YZ enabled
		std::atomic<bool>			img_acc_slice_yz_enabled;
		//! number of image in view XY to average
		std::atomic<uint>			img_acc_slice_xy_level;
		//! number of image in view XZ to average
		std::atomic<uint>			img_acc_slice_xz_level;
		//! number of image in view YZ to average
		std::atomic<uint>			img_acc_slice_yz_level;

		//! is p average enabled (average image over multiple depth index)
		std::atomic<bool>			p_accu_enabled;
		//! difference between p min and p max
		std::atomic<short>			p_acc_level;
		
		//! is x average in view YZ enabled (average of columns between both selected columns)
		std::atomic<bool>			x_accu_enabled;
		//! difference between x min and x max
		std::atomic<short>			x_acc_level;

		//! is y average in view XZ enabled (average of lines between both selected lines)
		std::atomic<bool>			y_accu_enabled;
		//! difference between y min and y max
		std::atomic<short>			y_acc_level;

		//! Enables the resizing of slice windows to have square pixels (according to their real size)
		std::atomic<bool>			square_pixel { false };

		//! Use Zernike polynomials instead of paraboloid for the lens
		std::atomic<bool>			zernike_enabled{ false };
		std::atomic<int>			zernike_m;
		std::atomic<int>			zernike_n;


		//! Display the raw interferogram when we are in hologram mode.
		std::atomic<bool>			raw_view { false };
		//! Enables the recording of the raw interferogram when we are in hologram mode.
		std::atomic<bool>			record_raw { false };

		//! Lock the zoom.
		std::atomic<bool>			locked_zoom{ false };

		//! Composite images
		//! \{
		std::atomic<ushort>		composite_p_red;
		std::atomic<ushort>		composite_p_blue;
		std::atomic<float>		weight_r;
		std::atomic<float>		weight_g;
		std::atomic<float>		weight_b;
		std::atomic<bool>		composite_auto_weights_;
		//! \}

		std::atomic<bool>			jitter_enabled_{ false };
		std::atomic<int>			jitter_slices_{ 7 };
		std::atomic<double>			jitter_factor_{ 1. };

		std::atomic<bool>			aberration_enabled_{ false };
		std::atomic<int>			aberration_slices_{ 8 };
		std::atomic<double>			aberration_factor_{ 1. };

		#pragma endregion
	};
}
