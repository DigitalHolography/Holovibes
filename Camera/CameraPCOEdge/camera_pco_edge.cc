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

#include "camera_pco_edge.hh"
#include <camera_exception.hh>
#include <boost/lexical_cast.hpp>

#include <PCO_err.h>
#include <sc2_defs.h>

namespace camera
{
	/* Load PCO_Signal data from ini file. */
	static void load_pco_signal_params(const boost::property_tree::ptree& pt,
		PCO_Signal& sig, const WORD sig_index)
	{
		// Field format for signal number X :   X_nameofthefield
		std::string prefix = "pco-edge." + boost::lexical_cast<std::string>(sig_index) + '_';

		sig.wEnabled = pt.get<WORD>(prefix + "state", sig.wEnabled);
		sig.wType = pt.get<WORD>(prefix + "type", sig.wType);
		sig.wPolarity = pt.get<WORD>(prefix + "polarity", sig.wPolarity);
		sig.wFilterSetting = pt.get<WORD>(prefix + "filter", sig.wFilterSetting);
		sig.wSelected = pt.get<WORD>(prefix + "subindex", sig.wSelected);

		/* The remaining dwParameter and dwSignalFunctionality are left untouched,
		** for tweaking them is not useful to us.
		*/
	}

	ICamera* new_camera_device()
	{
		return new CameraPCOEdge();
	}

	CameraPCOEdge::CameraPCOEdge()
		: CameraPCO("edge.ini", CAMERATYPE_PCO_EDGE_USB3)
	{
		name_ = "Edge 4.2 LT";

		load_default_params();
		if (ini_file_is_open())
			load_ini_params();

		if (ini_file_is_open())
			ini_file_.close();
	}

	CameraPCOEdge::~CameraPCOEdge()
	{
	}

	void CameraPCOEdge::load_default_params()
	{
		//FrameDescriptor fd = get_frame_descriptor();
		/* Various camera parameters. */
		exposure_time_ = 0.024f;

		triggermode_ = 0;

		framerate_ = 30 * 1e3;
		framerate_mode_ = 1;

		binning_ = 1;

		p0_x_ = 1;
		p0_y_ = 1;
		p1_x_ = 2048;
		p1_y_ = 2048;

		pixel_rate_ = 110 * 1e6;

		conversion_factor_ = 46;

		timeouts_[0] = timeouts_[1] = timeouts_[2] = 50;

		/* Hardware signals (io_1_conf, etc.) are not initialized manually, because
		** default settings are sufficient for the camera to work with them.
		** Only settings in the ini file shall be taken into account, later.
		*/
		io_0_conf.wSize = sizeof(PCO_Signal);
		io_1_conf.wSize = sizeof(PCO_Signal);
		io_2_conf.wSize = sizeof(PCO_Signal);
		io_3_conf.wSize = sizeof(PCO_Signal);

		/* Fill frame descriptor const values. */
		desc_.depth = 2.f;
		desc_.byteEndian = Endianness::LittleEndian;
		pixel_size_ = 6.45f;
		desc_.width = 2048;
		desc_.height = 2048;
	}

	void CameraPCOEdge::load_ini_params()
	{
		//FrameDescriptor fd = get_frame_descriptor();
		const boost::property_tree::ptree& pt = get_ini_pt();

		exposure_time_ = pt.get<float>("pco-edge.exposure_time", exposure_time_);

		triggermode_ = pt.get<WORD>("pco-edge.trigger_mode", triggermode_);

		framerate_ = pt.get<DWORD>("pco-edge.framerate", framerate_) * 1e3;
		framerate_mode_ = pt.get<WORD>("pco-edge.framerate_mode", framerate_mode_);

		binning_ = pt.get<WORD>("pco-edge.binning", binning_);
		// Updating frame descriptor's dimensions accordingly.
		pixel_size_ *= binning_;
		desc_.width /= binning_;
		desc_.height /= binning_;

		{
			// Making sure ROI settings are valid.

			WORD tmp_p0_x = pt.get<WORD>("pco-edge.roi_x", p0_x_);
			WORD tmp_p0_y = pt.get<WORD>("pco-edge.roi_y", p0_y_);
			WORD tmp_p1_x = pt.get<WORD>("pco-edge.roi_width", p1_x_) + tmp_p0_x - 1;
			WORD tmp_p1_y = pt.get<WORD>("pco-edge.roi_height", p1_y_) + tmp_p0_y - 1;

			if (tmp_p0_x < tmp_p1_x &&
				tmp_p0_y < tmp_p1_y &&
				tmp_p0_x <= desc_.width &&
				tmp_p0_y <= desc_.height &&
				tmp_p1_x <= desc_.width &&
				tmp_p1_y <= desc_.height)
			{
				p0_x_ = tmp_p0_x;
				p0_y_ = tmp_p0_y;
				p1_x_ = tmp_p1_x;
				p1_y_ = tmp_p1_y;

				// Don't forget to update frame descriptor
				desc_.width = p1_x_ - p0_x_ + 1;
				desc_.height = p1_y_ - p0_y_ + 1;
			}
			else
				std::cerr << "[CAMERA] Invalid ROI settings, ignoring ROI." << std::endl;
		}

		pixel_rate_ = pt.get<DWORD>("pco-edge.pixel_rate", pixel_rate_) * 1e6;

		conversion_factor_ = static_cast<WORD>(
			pt.get<float>("pco-edge.conversion_factor", conversion_factor_) * 100);

		timeouts_[0] = pt.get<unsigned int>("pco-edge.timeout_command", timeouts_[0]) * 1e3;
		timeouts_[1] = pt.get<unsigned int>("pco-edge.timeout_img_acq", timeouts_[1]) * 1e3;

		load_pco_signal_params(pt, io_0_conf, 0);
		load_pco_signal_params(pt, io_1_conf, 1);
		load_pco_signal_params(pt, io_2_conf, 2);
		load_pco_signal_params(pt, io_3_conf, 3);
	}

	void CameraPCOEdge::bind_params()
	{
		int status = PCO_NOERROR;
		status |= PCO_ResetSettingsToDefault(device_);

		status |= PCO_SetTimeouts(device_, timeouts_, 2);

		status |= PCO_SetSensorFormat(device_, 0);

		status |= PCO_SetConversionFactor(device_, conversion_factor_);

		status |= PCO_SetBinning(device_, binning_, binning_);

		status |= PCO_SetROI(device_, p0_x_, p0_y_, p1_x_, p1_y_);

		status |= PCO_SetPixelRate(device_, pixel_rate_);

		status |= PCO_SetTriggerMode(device_, triggermode_);

		status |= PCO_SetNoiseFilterMode(device_, 0);

		{
			/* Convert exposure time in milliseconds. */
			float tmp_exp_time = exposure_time_;
			tmp_exp_time *= 1e3;

			/* base_time : 0x0002 = ms, 0x0001 = us, 0x0000 = ns */
			WORD base_time;

			// Why doing this?
			for (base_time = 0x0002; base_time > 0 && tmp_exp_time < 1.0f; --base_time)
				tmp_exp_time *= 1e3;

			status |= PCO_SetDelayExposureTime(device_, 0, static_cast<DWORD>(tmp_exp_time), 0, base_time);
		}

		{
			WORD fps_change_status; // Knowing if some value was trimmed. Currently unused.
			DWORD tmp_exp_time = static_cast<DWORD>(exposure_time_ * 1e9); // SDK requires exp. time in ns

			status |= PCO_SetFrameRate(device_, &fps_change_status, framerate_mode_, &framerate_, &tmp_exp_time);

			exposure_time_ = static_cast<float>(tmp_exp_time* 1e-9); // Convert back exp. time to seconds
		}

		{
			status |= PCO_SetHWIOSignal(device_, 0, &io_0_conf);
			status |= PCO_SetHWIOSignal(device_, 1, &io_1_conf);
			status |= PCO_SetHWIOSignal(device_, 2, &io_2_conf);
			status |= PCO_SetHWIOSignal(device_, 3, &io_3_conf);
		}

		if (status != PCO_NOERROR)
			throw CameraException(CameraException::CANT_SET_CONFIG);
	}
}