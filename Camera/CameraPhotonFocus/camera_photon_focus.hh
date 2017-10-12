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

		// Convert string read from config file to PvPixelType enum. Set default value if input is empty
		PvPixelType get_pixel_type(std::string input);

		//Debugging function to display information about the image. Debug purpose
		void display_image(PvImage *image);

		/* \brief Display every parameter of a gen parameter array. Debug purpose
		*  \param aArray GenParameterArray to display
		*  \param Param Limit the display to a category of parameters
		*/
		void DumpGenParameterArray(PvGenParameterArray *aArray, std::string Param = "");

	private:
		PvResult result_; // containing the result status of each function called in the photonfocus sdk
		PvDevice device_; // The connected camera
		PvGenParameterArray *device_params_; // Parameters of the device, used to control the streaming
		PvStream stream_; // The stream linked with the device. We need it as attribute to close it properly.
		std::unique_ptr<PvPipeline> pipeline_; // Wrapper around PvStream to control it more easily.

		std::unique_ptr<PvUInt8[]> output_image_; // Output buffer

		/* Image Format */
		unsigned short offset_x_; // x coordinate of the top-left corner of ROI
		unsigned short offset_y_; // y coordinate of the top-left corner of ROI
		PvPixelType pixel_type_; // Encoding format of pixels

		/* Acquisition Control */
		bool frame_rate_enable_; // enables constant frame rate
		float frame_rate_; // fps. Computed automatically if frame_rate_enable_ is not set
	};
}