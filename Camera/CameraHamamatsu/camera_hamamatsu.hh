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
* Camera Hamamatsu C11440 */
#pragma once

#include <camera.hh>
#include "camera_exception.hh"

#include "dcamapi4.h"
#include "dcamprop.h"

namespace camera
{
	class CameraHamamatsu : public Camera
	{
	public:
		CameraHamamatsu();

		virtual ~CameraHamamatsu()
		{}

		virtual void init_camera() override;
		virtual void start_acquisition() override;
		virtual void stop_acquisition() override;
		virtual void shutdown_camera() override;
		virtual CapturedFramesDescriptor get_frames() override;

	private:
		virtual void load_ini_params() override;
		virtual void load_default_params() override;
		virtual void bind_params() override;

		void retrieve_camera_name();
		void retrieve_pixel_depth();
		void allocate_host_frame_buffer();

		void get_event_waiter_handle();

		void set_frame_acq_info();
		void set_wait_info();

		std::unique_ptr<unsigned short[]> output_frame_;

		HDCAM hdcam_; //Camera handle
		HDCAMWAIT hwait_; //Event waiter handle (will typically wait for the FRAME_READY event)
		/*! \brief start position x axis of the region of interest */
		long srcox_;
		/*! \brief start position y axis of the region of interest */
		long srcoy_;
		unsigned short binning_; // 1, 2, 4
		bool ext_trig_;
		int32 circ_buffer_frame_count_;
		_DCAMPROPMODEVALUE trig_mode_; // NORMAL, START
		_DCAMPROPMODEVALUE trig_connector_; // INTERFACE, BNC
		_DCAMPROPMODEVALUE trig_polarity_; // POSITIVE, NEGATIVE
		_DCAMPROPMODEVALUE readoutspeed_; // SLOWEST, FASTEST
		_DCAMPROPMODEVALUE trig_active_; // EDGE, LEVEL,

		static constexpr unsigned short MAX_WIDTH = 2304;
		static constexpr unsigned short MAX_HEIGHT = 2304;

		//Structures needed to acquire frames
		//Will be passed as arguments to the API functions
		DCAMBUF_FRAME dcam_frame_acq_info_;
		DCAMWAIT_START dcam_wait_info_;
	};
}
