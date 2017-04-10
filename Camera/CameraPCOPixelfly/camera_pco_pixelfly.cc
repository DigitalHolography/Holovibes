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

#include "camera_pco_pixelfly.hh"
#include <camera_exception.hh>

#include <PCO_err.h>
#include <sc2_defs.h>

namespace camera
{
	ICamera* new_camera_device()
	{
		return new CameraPCOPixelfly();
	}

	CameraPCOPixelfly::CameraPCOPixelfly()
		: CameraPCO("pixelfly.ini", CAMERATYPE_PCO_USBPIXELFLY)
		, squared_buffer_(new WORD[4194304]) // 2048 * 2048
	{
		name_ = "pixelfly";

		load_default_params();
		if (ini_file_is_open())
			load_ini_params();

		if (ini_file_is_open())
			ini_file_.close();
		unsigned int frame_size = 4194304; // 2048 * 2048
		for (unsigned int i = 0; i < frame_size; ++i)
			squared_buffer_[i] = 0;
	}

	CameraPCOPixelfly::~CameraPCOPixelfly()
	{
		delete squared_buffer_;
	}

	void* CameraPCOPixelfly::get_frame()
	{
		WORD actual_res_x = get_actual_res_x();
		WORD actual_res_y = get_actual_res_y();

		WORD* buffer = static_cast<WORD*>(CameraPCO::get_frame());

		unsigned int squared_buffer_offset = 0;
		for (WORD y = 0; y < actual_res_y; ++y)
		{
			for (WORD x = 0; x < actual_res_x; ++x)
				squared_buffer_[squared_buffer_offset + x] = buffer[y * actual_res_x + x];

			/* Add size of a squared buffer line. */
			squared_buffer_offset += 2048;
		}

		return squared_buffer_;
	}

	void CameraPCOPixelfly::load_default_params()
	{
		exposure_time_ = 0.050f;
		extended_sensor_format_ = false;
		pixel_rate_ = 12;
		binning_enabled_ = false;
		ir_sensitivity_ = false;

		/* Fill frame descriptor const values. */
		desc_.depth = 2;
		desc_.endianness = LITTLE_ENDIAN;
		desc_.pixel_size = 6.45f;
		desc_.width = 2048;
		desc_.height = 2048;
	}

	void CameraPCOPixelfly::load_ini_params()
	{
		const boost::property_tree::ptree& pt = get_ini_pt();

		exposure_time_ = pt.get<float>("pco-pixelfly.exposure_time", exposure_time_);
		extended_sensor_format_ = pt.get<bool>("pco-pixelfly.extended_sensor_format", extended_sensor_format_);
		pixel_rate_ = pt.get<unsigned int>("pco-pixelfly.pixel_rate", pixel_rate_);
		if (pixel_rate_ != 12 && pixel_rate_ != 25)
			pixel_rate_ = 12;
		binning_enabled_ = pt.get<bool>("pco-pixelfly.binning", binning_enabled_);
		ir_sensitivity_ = pt.get<bool>("pco-pixelfly.ir_sensitivity", ir_sensitivity_);
	}

	void CameraPCOPixelfly::bind_params()
	{
		int status = PCO_NOERROR;

		status |= PCO_SetSensorFormat(device_, extended_sensor_format_);
		status |= PCO_SetPixelRate(device_, static_cast<DWORD>(pixel_rate_ * 1e6));
		status |= PCO_SetIRSensitivity(device_, ir_sensitivity_);
		{
			WORD binning = (binning_enabled_ ? 2 : 1);
			desc_.pixel_size *= binning;
			status |= PCO_SetBinning(device_, binning, binning);
		}
		{
			/* Convert exposure time in milliseconds. */
			exposure_time_ *= 1e3;

			/* base_time : 0x0002 = ms, 0x0001 = us, 0x0000 = ns */
			WORD base_time;

			for (base_time = 0x0002; base_time > 0 && exposure_time_ < 1.0f; --base_time)
				exposure_time_ *= 1e3;

			status |= PCO_SetDelayExposureTime(device_, 0, static_cast<DWORD>(exposure_time_), 0, base_time);
		}

		if (status != PCO_NOERROR)
			throw CameraException(CameraException::CANT_SET_CONFIG);
	}
}