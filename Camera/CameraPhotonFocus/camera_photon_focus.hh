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

#include <camera.hh>

#include <PvDevice.h>
#include <PvPipeline.h>
#include <PvBuffer.h>
#include <PvStream.h>
#include <PvSystem.h>
#include <PvInterface.h>

#include "camera_exception.hh"

namespace camera
{
	class CameraPhotonFocus : public Camera
	{
	public:
		CameraPhotonFocus();

		virtual ~CameraPhotonFocus()
		{}

		virtual void init_camera() override;
		virtual void start_acquisition() override;
		virtual void stop_acquisition() override;
		virtual void shutdown_camera() override;
		virtual void *get_frame() override;

	private:
		virtual void load_ini_params() override;
		virtual void load_default_params() override;
		virtual void bind_params() override;

		//Debugging function to display information about the frame captured, like padding or offset
		void display_image(PvImage *image);

		bool DumpGenParameterArray(PvGenParameterArray *aArray);

	private:
		PvResult result_; // containing the result status of each function called in the photonfocus sdk
		PvDevice device_; // The connected camera
		std::unique_ptr<PvGenParameterArray> device_params_; // Parameters of the device, used to control the streaming
		PvStream stream_; // The stream linked with the device. We need it as attribute to close it properly.
		std::unique_ptr<PvPipeline> pipeline_; // Wrapper around PvStream to control it more easily.

		unsigned short offset_x_;
		unsigned short offset_y_;
		float frame_rate_;
	};
}