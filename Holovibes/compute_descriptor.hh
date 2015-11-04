#pragma once

# include <atomic>

# include "observable.hh"
# include "geometry.hh"

namespace holovibes
{
  const static std::string version = "v1.2.0";

	/*! \brief Contains compute parameters.
	 *
	 * Theses parameters will be used when the pipeline is refresh.
	 * It defines parameters for FFT, lens (Fresnel transforms ...),
	 * post-processing (contrast, shift_corners, log scale).
	 *
	 * The class use the *Observer* design pattern instead of the signal
	 * mechanism of Qt because classes in the namespace holovibes are
	 * independent of GUI or CLI implementations. So that, the code remains
	 * reusable.
	 *
	 * This class contains std::atomic fields to avoid concurrent access between
	 * the pipeline and the GUI.
	 */
	struct ComputeDescriptor : public Observable
	{
		/*! \brief Select hologram methods. */
		enum fft_algorithm
		{
			FFT1,
			FFT2,
			STFT,
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
		};

		/*! \brief ComputeDescriptor constructor
		 *
		 * Initialize the compute descriptor to default values of computation. */
		ComputeDescriptor()
			: Observable()
			, algorithm(FFT1)
			, nsamples(2)
			, pindex()
			, lambda(532e-9f)
			, zdistance(1.50f)
			, view_mode(MODULUS)
			, log_scale_enabled(false)
			, shift_corners_enabled(true)
			, contrast_enabled(false)
			, vibrometry_enabled(false)
			, contrast_min(1)
			, contrast_max(65535)
			, vibrometry_q()
		{
			pindex = 0;
			vibrometry_q = 0;
		}

		/*! \brief Assignment operator
		 *
		 * The assignment operator is explicitely defined because std::atomic type
		 * does not allow to generate assignments operator automatically. */
		ComputeDescriptor& operator=(const ComputeDescriptor& cd);

		/*! \brief Check is parameters are consistent.
		 *
		 * \note This method is unused because CLI/GUI already implements the
		 * checking. But it could be a good idea to finally implement it, to make
		 * the checking method in common. */
		void sanity_check();

		/*! Hologram algorithm. */
		std::atomic<enum fft_algorithm> algorithm;
		/*! Number of samples in which apply the fft on. */
		std::atomic<unsigned short> nsamples;
		/*! p-th output component to show. */
		std::atomic_ushort pindex;
		/*! Lambda in meters. */
		std::atomic<float> lambda;
		/*! Sensor-to-object distance. */
		std::atomic<float> zdistance;
		/*! Complex to float method. */
		std::atomic<enum complex_view_mode> view_mode;
		/*! Is log scale post-processing enabled. */
		std::atomic<bool> log_scale_enabled;
		/*! Is FFT shift corners post-processing enabled. */
		std::atomic<bool> shift_corners_enabled;
		/*! Is manual contrast post-processing enabled. */
		std::atomic<bool> contrast_enabled;
		/*! Is vibrometry method enabled. */
		std::atomic<bool> vibrometry_enabled;
		/*! Contrast minimal range value. */
		std::atomic<float> contrast_min;
		/*! Contrast maximal range value. */
		std::atomic<float> contrast_max;
		/*! q-th output component of FFT to use with vibrometry. */
		std::atomic<unsigned short> vibrometry_q;

		/*! Average mode signal zone */
		std::atomic<Rectangle> signal_zone;
		/*! Average mode noise zone */
		std::atomic<Rectangle> noise_zone;

		/*! Z minimal range for autofocus. */
		std::atomic<float> autofocus_z_min;
		/*! Z maximal range for autofocus. */
		std::atomic<float> autofocus_z_max;
		/*! Number of points of autofocus between the z range. */
		std::atomic<unsigned int> autofocus_z_div;
		/*! Number of iterations of autofocus between the z range. */
    std::atomic<unsigned int> autofocus_z_iter;
    /*! Selected zone in which apply the autofocus algorithm. */
    std::atomic<Rectangle> autofocus_zone;
    /*! Selected zone in which apply the stft algorithm. */
    std::atomic<Rectangle> stft_roi_zone;
	};
}