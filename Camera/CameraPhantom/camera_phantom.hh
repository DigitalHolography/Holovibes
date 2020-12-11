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

#include <EGrabber.h>

#include "camera.hh"
#include "camera_exception.hh"

namespace camera
{
	class EuresysCustomGrabber : public Euresys::EGrabberCallbackOnDemand
	{
	  public:
		EuresysCustomGrabber(Euresys::EGenTL &gentl)
			: Euresys::EGrabberCallbackOnDemand(gentl)
		{
		}

		~EuresysCustomGrabber() { shutdown(); }

		char *last_device_ptr_;

	  private:
		virtual void onNewBufferEvent(const Euresys::NewBufferData &data)
		{
			Euresys::ScopedBuffer buffer(*this, data);
			last_device_ptr_ = (char *) buffer.getUserPointer();
		}
	};

	class CameraPhantom : public Camera
	{
	public:
		CameraPhantom();
		virtual ~CameraPhantom() {}

		virtual void init_camera() override;
		virtual void start_acquisition() override;
		virtual void stop_acquisition() override;
		virtual void shutdown_camera() override;
		virtual CapturedFramesDescriptor get_frames() override;

	private:
		virtual void load_ini_params() override;
		virtual void load_default_params() override;
		virtual void bind_params() override;

		std::unique_ptr<Euresys::EGenTL> gentl_;
		std::unique_ptr<EuresysCustomGrabber> grabber_;
		std::vector<unsigned char *> buffers_;

		unsigned int nb_buffers_;
		unsigned int width_;
		unsigned int height_;
		unsigned int roi_x_;
		unsigned int roi_y_;
		float frame_period_;
		float exposure_time_;
	};
}