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

#include <iostream>
#include <chrono>
#include <stdexcept>

#include "camera_hamamatsu.hh"

namespace camera
{
	CameraHamamatsu::CameraHamamatsu()
		: Camera("hamamatsu.ini")
	{
		name_ = "MISSING NAME";

		load_default_params();

		if (ini_file_is_open()) {
			load_ini_params();
			ini_file_.close();
		}
	}

	void CameraHamamatsu::init_camera()
	{
		char	buf[256];
		int32	nDevice;

		// initialize DCAM-API
		if (!dcam_init(NULL, &nDevice, NULL) || nDevice <= 0)
			throw CameraException(CameraException::NOT_CONNECTED);

		// show all camera information by text
		for (int32 iDevice = 0; iDevice < nDevice; iDevice++)
		{
			dcam_getmodelinfo(iDevice, DCAM_IDSTR_VENDOR, buf, sizeof(buf));
			if (strcmp(buf, "Hamamatsu"))
				continue;

			dcam_getmodelinfo(iDevice, DCAM_IDSTR_MODEL, buf, sizeof(buf));

			//Using the std::string(const char *, size_t) constructor would not stop at the null bytes
			//This should not be needed, but it could happen in some unlucky setting
			buf[sizeof(buf) - 1] = '\0';
			name_ = buf;

			std::cout << "Connecting to " << name_ << std::endl;
			if (dcam_open(&hdcam_, iDevice, NULL))
			{
				bind_params();

				double bits_per_channel;
				dcam_getpropertyvalue(hdcam_, DCAM_IDPROP_BITSPERCHANNEL, &bits_per_channel);
				desc_.depth = bits_per_channel / 8;

				output_frame_ = std::make_unique<unsigned short[]>(desc_.frame_res());
				memset(output_frame_.get(), 0, desc_.frame_res());
				return; // SUCCESS
			}
			else {
				dcam_uninit(NULL, NULL);
				throw CameraException(CameraException::NOT_CONNECTED);
			}
		}
		throw CameraException(CameraException::NOT_CONNECTED);
	}

	void CameraHamamatsu::start_acquisition()
	{
		if (!dcam_precapture(hdcam_, DCAM_CAPTUREMODE_SEQUENCE))
			throw CameraException(CameraException::CANT_START_ACQUISITION);

		int32 frame_count = 1;
		// allocate capturing buffer
		if (!dcam_allocframe(hdcam_, frame_count))
			throw CameraException(CameraException::CANT_START_ACQUISITION);

		// start capturing
		if (!dcam_capture(hdcam_))
			throw CameraException(CameraException::CANT_START_ACQUISITION);
	}

	void CameraHamamatsu::stop_acquisition()
	{
		// stop capturing
		dcam_idle(hdcam_);
		// release capturing buffer
		dcam_freeframe(hdcam_);
	}

	void CameraHamamatsu::shutdown_camera()
	{
		// close HDCAM handle
		dcam_close(hdcam_);

		// terminate DCAM-API
		dcam_uninit(NULL, NULL);
	}

	void* CameraHamamatsu::get_frame()
	{
		int32 sRow;
		unsigned short* src;

		/*if (!dcam_firetrigger(hdcam_)) {
			//throw CameraException::CANT_GET_FRAME;
			std::cerr << "Cant fire trigger" << std::endl;
		}*/
		//_DWORD	dw = DCAM_EVENT_FRAMEEND;

		long	err = dcam_getlasterror(hdcam_);
		if (err == DCAMERR_TIMEOUT)
			throw CameraException(CameraException::CANT_GET_FRAME);

		if (dcam_lockdata(hdcam_, (void**)&(src), &sRow, -1)) {
			
			//WORD*	dsttopleft = output_frame_.get();

			//long srcwidth = desc_.width, srcheight = desc_.height;
			//const BYTE* lut = nullptr;
			//const WORD* srctopleft = src;
			//long srcrowbytes = desc_.width;

			//copybits_bw16(dsttopleft, desc_.width, lut
			//	, (const WORD*)srctopleft, srcrowbytes
			//	, 0/*srcox_*/, 0/*srcoy_*/, srcwidth, srcheight);

			// The above looks like software level ROI management

			memcpy(output_frame_.get(), src, desc_.frame_size());

			dcam_unlockdata(hdcam_);
		}
		return output_frame_.get();
	}

	/*long CameraHamamatsu::copybits_bw16(WORD* dsttopleft, long dstrowpix, const BYTE* lut
		, const WORD* srctopleft, long srcrowpix
		, long srcox, long srcoy, long srcwidth, long srcheight)
	{
		long	lines = 0;
		const WORD*	src = srctopleft + srcrowpix * srcoy + srcox;
		WORD* dst = dsttopleft;

		int	x, y;
		for (y = srcheight; y-- > 0; )
		{
			const WORD*	s = (const WORD*)src;
			WORD*	d = dst;

			for (x = srcwidth; x-- > 0; )
				*d++ = *s++; // lut[*s++];

			src += srcrowpix;
			dst += dstrowpix;
			lines++;
		}

		return lines;
	}*/

	void CameraHamamatsu::load_default_params()
	{
		desc_.width = 2048;
		desc_.height = 2048;
		desc_.depth = 2;
		desc_.byteEndian = Endianness::LittleEndian;

		pixel_size_ = 6.5f;

		exposure_time_ = 50000;

		srcox_ = 0;
		srcoy_ = 0;

		binning_ = 1;

		ext_trig_ = false;
		trig_connector_ = DCAMPROP_TRIGGER_CONNECTOR__BNC;
		trig_polarity_ = DCAMPROP_TRIGGERPOLARITY__NEGATIVE;

		readoutspeed_ = DCAMPROP_READOUTSPEED__FASTEST;
	}

	void CameraHamamatsu::load_ini_params()
	{
		/* Use the default value in case of fail. */
		const boost::property_tree::ptree& pt = get_ini_pt();

		name_ = pt.get<std::string>("hamamatsu.name", name_);

		desc_.width = pt.get<unsigned short>("hamamatsu.roi_width", desc_.width);
		desc_.height = pt.get<unsigned short>("hamamatsu.roi_height", desc_.height);
		srcox_ = pt.get<long>("hamamatsu.roi_startx", srcox_);
		srcoy_ = pt.get<long>("hamamatsu.roi_starty", srcoy_);

		exposure_time_ = pt.get<float>("hamamatsu.exposure_time", exposure_time_);

		binning_ = pt.get<unsigned short>("hamamatsu.binning", binning_);

		ext_trig_ = pt.get<bool>("hamamatsu.ext_trig", ext_trig_);

		std::string trig_connector = pt.get<std::string>("hamamatsu.trig_connector", "");
		if (trig_connector == "INTERFACE")
			trig_connector_ = DCAMPROP_TRIGGER_CONNECTOR__INTERFACE;
		else if (trig_connector == "BNC")
			trig_connector_ = DCAMPROP_TRIGGER_CONNECTOR__BNC;

		std::string trig_polarity = pt.get<std::string>("hamamatsu.trig_polarity", "");
		if (trig_polarity == "POSITIVE")
			trig_polarity_ = DCAMPROP_TRIGGERPOLARITY__POSITIVE;
		else if (trig_polarity == "NEGATIVE")
			trig_polarity_ = DCAMPROP_TRIGGERPOLARITY__NEGATIVE;
		
		std::string readoutspeed = pt.get<std::string>("hamamatsu.readoutspeed", "");
		if (readoutspeed == "SLOWEST")
			readoutspeed_ = DCAMPROP_READOUTSPEED__SLOWEST;
		else if (readoutspeed == "FASTEST")
			readoutspeed_ = DCAMPROP_READOUTSPEED__FASTEST;
	}

	void CameraHamamatsu::bind_params()
	{
		if (desc_.width != 2048 || desc_.height != 2048) // SUBARRAY
		{
			dcam_setpropertyvalue(hdcam_, DCAM_IDPROP_SUBARRAYMODE, DCAMPROP_MODE__ON);
			dcam_setpropertyvalue(hdcam_, DCAM_IDPROP_SUBARRAYHSIZE, desc_.width * binning_);
			dcam_setpropertyvalue(hdcam_, DCAM_IDPROP_SUBARRAYVSIZE, desc_.height * binning_);
			dcam_setpropertyvalue(hdcam_, DCAM_IDPROP_SUBARRAYHPOS, srcox_);
			dcam_setpropertyvalue(hdcam_, DCAM_IDPROP_SUBARRAYVPOS, srcoy_);
		}

		if (!dcam_setexposuretime(hdcam_, exposure_time_ / 1E6))
			throw CameraException(CameraException::CANT_SET_CONFIG);

		if (!dcam_setbinning(hdcam_, binning_))
			throw CameraException(CameraException::CANT_SET_CONFIG);

		if (!dcam_settriggermode(hdcam_, ext_trig_ ? DCAM_TRIGMODE_EDGE: DCAM_TRIGMODE_INTERNAL))
			throw CameraException(CameraException::CANT_SET_CONFIG);
		if (!dcam_setpropertyvalue(hdcam_, DCAM_IDPROP_TRIGGER_CONNECTOR, trig_connector_))
			throw CameraException(CameraException::CANT_SET_CONFIG);
		if (!dcam_setpropertyvalue(hdcam_, DCAM_IDPROP_TRIGGERPOLARITY, trig_polarity_))
			throw CameraException(CameraException::CANT_SET_CONFIG);
		if (!dcam_setpropertyvalue(hdcam_, DCAM_IDPROP_READOUTSPEED, readoutspeed_))
			throw CameraException(CameraException::CANT_SET_CONFIG);
	}

	ICamera* new_camera_device()
	{
		return new CameraHamamatsu();
	}

}