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

#include "camera_photon_focus.hh"

namespace camera
{
	CameraPhotonFocus::CameraPhotonFocus()
		: Camera("photonfocus.ini")
		, device_params_(nullptr)
	{
		name_ = "MV1-D1312IE-100-G2-12";

		load_default_params();

		if (ini_file_is_open())
			load_ini_params();

		if (ini_file_is_open())
			ini_file_.close();
	}

	void CameraPhotonFocus::init_camera()
	{
		//
		PvDeviceInfo *lDeviceInfo = nullptr;

		// Create an GEV system and an interface.
		PvSystem lSystem;

		// Find all GEV Devices on the network.
		lSystem.SetDetectionTimeout(1000);
		result_ = lSystem.Find();
		if (!result_.IsOK())
			throw CameraException(CameraException::NOT_CONNECTED);

		PvUInt32 lInterfaceCount = lSystem.GetInterfaceCount();

		//Iterate over every interface to find the device. We must do this because sometimes, it considers that the network card is a GEV device.
		for (PvUInt32 x = 0; x < lInterfaceCount; x++)
		{
			// get pointer to each of interface
			PvInterface * lInterface = lSystem.GetInterface(x);

			PvUInt32 lDeviceCount = lInterface->GetDeviceCount();

			for (PvUInt32 y = 0; y < lDeviceCount; y++)
			{
				PvDeviceInfo *device_info_tmp = lInterface->GetDeviceInfo(y);
				if (device_info_tmp->GetModel().GetAscii() == name_)
					lDeviceInfo = device_info_tmp;
			}
		}

		// Connect to the last GEV Device found.
		if (lDeviceInfo)
		{
			std::cout << "Connecting to " << lDeviceInfo->GetModel().GetAscii() << std::endl;
			result_ = device_.Connect(lDeviceInfo);
			if (!result_.IsOK())
				throw CameraException(CameraException::NOT_INITIALIZED);
		}
		else
			throw CameraException(CameraException::NOT_CONNECTED);

		// Retrieving information about the camera model
		device_params_ = device_.GetGenParameters();

		// Automatically find and set the optimal packet size
		result_ = device_.NegotiatePacketSize();
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_START_ACQUISITION);

		// Open a communication with the device.
		result_ = stream_.Open(lDeviceInfo->GetIPAddress());
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_START_ACQUISITION);

		bind_params();

		output_image_ = std::make_unique<PvUInt8[]>(desc_.frame_size());
	}

	void CameraPhotonFocus::start_acquisition()
	{
		// TLParamsLocked is optional but when present, it MUST be set to 1
		// before sending the AcquisitionStart command
		device_params_->SetIntegerValue("TLParamsLocked", 1);

		//DumpGenParameterArray(device_params_);

		result_ = device_params_->ExecuteCommand("AcquisitionStart");
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_START_ACQUISITION);
		
		for (int i = 0; i < 20; ++i) {
			PvBuffer *buffer = nullptr;
			PvResult  operation_result;
			result_ = pipeline_->RetrieveNextBuffer(&buffer, FRAME_TIMEOUT, &operation_result);
			if (!result_.IsOK() || !operation_result.IsOK())
				std::cout << result_.GetCode() << std::endl;
			pipeline_->ReleaseBuffer(buffer);
		}
	}

	void CameraPhotonFocus::stop_acquisition()
	{
		result_ = device_params_->ExecuteCommand("AcquisitionStop");
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_STOP_ACQUISITION);

		// If present reset TLParamsLocked to 0. Must be done AFTER the 
		// streaming has been stopped
		device_params_->SetIntegerValue("TLParamsLocked", 0);
	}

	void CameraPhotonFocus::shutdown_camera()
	{
		pipeline_->Stop();
		stream_.Close();
		device_.Disconnect();
	}

	void* CameraPhotonFocus::get_frame()
	{
		// Retrieve next buffer		
		PvBuffer *buffer = nullptr;
		PvResult  operation_result;

		result_ = pipeline_->RetrieveNextBuffer(&buffer, FRAME_TIMEOUT, &operation_result);
		
		// Connection problem or Timeout
		if (result_.IsOK()) {
			if (operation_result.IsOK()) {
				//Processing buffer to retrieve a frame
				if (buffer->GetPayloadType() == PvPayloadTypeImage)
				{
					PvImage *image = buffer->GetImage();
					unsigned char *raw_buffer = image->GetDataPointer();
					memcpy(output_image_.get(), raw_buffer, desc_.frame_size());
				}
			}

			pipeline_->ReleaseBuffer(buffer);
			if (!operation_result.IsOK())
				throw CameraException(CameraException::CANT_GET_FRAME);
		}
		else
			throw CameraException(CameraException::CANT_GET_FRAME);
		//Problem related to the stream initialization
		return output_image_.get();
	}

	void CameraPhotonFocus::load_default_params()
	{
		desc_.width = 1024;
		desc_.height = 1024;
		desc_.depth = 1.0f;
		desc_.byteEndian = Endianness::LittleEndian;

		pixel_size_ = 8.0f;

		pixel_type_ = PvPixelType::PvPixelMono8;

		exposure_time_ = 5000;

		frame_rate_enable_ = true;
		frame_rate_ = 60.0f;


	}

	void CameraPhotonFocus::load_ini_params()
	{
		/* Use the default value in case of fail. */
		const boost::property_tree::ptree& pt = get_ini_pt();

		name_ = pt.get<std::string>("photonfocus.name", name_);

		desc_.width = pt.get<unsigned short>("photonfocus.roi_width", desc_.width);
		desc_.height = pt.get<unsigned short>("photonfocus.roi_height", desc_.height);

		offset_x_ = pt.get<unsigned short>("photonfocus.roi_startx", offset_x_);
		offset_y_ = pt.get<unsigned short>("photonfocus.roi_starty", offset_y_);

		pixel_type_ = get_pixel_type(pt.get<std::string>("photonfocus.pixel_type", ""));

		exposure_time_ = pt.get<float>("photonfocus.exposure_time", exposure_time_);

		frame_rate_enable_ = pt.get<bool>("photonfocus.frame_rate_enable", frame_rate_enable_);
		if (frame_rate_enable_)
			frame_rate_ = pt.get<float>("photonfocus.frame_rate", frame_rate_);
	}

	void CameraPhotonFocus::bind_params()
	{
		/* Setting device configuration */

		result_ = device_params_->SetIntegerValue("Width", desc_.width);
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_SET_CONFIG);

		result_ = device_params_->SetIntegerValue("Height", desc_.height);
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_SET_CONFIG);

		result_ = device_params_->SetIntegerValue("OffsetX", offset_x_);
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_SET_CONFIG);

		result_ = device_params_->SetIntegerValue("OffsetY", offset_y_);
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_SET_CONFIG);

		result_ = device_params_->SetEnumValue("PixelFormat", pixel_type_);
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_SET_CONFIG);

		// Setting exposure time BEFORE frame rate, since exposure time change the allowed boundaries of the frame rate value.
		result_ = device_params_->SetFloatValue("ExposureTime", exposure_time_);
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_SET_CONFIG);

		result_ = device_params_->SetBooleanValue("AcquisitionFrameRateEnable", frame_rate_enable_);
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_SET_CONFIG);

		if (frame_rate_enable_)
		{
			result_ = device_params_->SetFloatValue("AcquisitionFrameRate", frame_rate_);
			if (!result_.IsOK())
				throw CameraException(CameraException::CANT_SET_CONFIG);
		}

		pipeline_ = std::make_unique<PvPipeline>(&stream_);

		// Retrieving the maximum buffer size of the camera model
		PvInt64 lSize = 0;
		result_ = device_params_->GetIntegerValue("PayloadSize", lSize);
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_START_ACQUISITION);

		// Set the Buffer size and the Buffer count
		//pipeline_->SetBufferSize(static_cast<PvUInt32>(lSize));
		//pipeline_->SetBufferSize(PvBuffer::GetRequiredSize());

		result_ = pipeline_->SetBufferCount(16); // Increase for high frame rate without missing block IDs
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_START_ACQUISITION);

		// Have to set the Device IP destination to the Stream
		result_ = device_.SetStreamDestination(stream_.GetLocalIPAddress(), stream_.GetLocalPort());
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_START_ACQUISITION);

		result_ = pipeline_->Start();
		if (!result_.IsOK())
			throw CameraException(CameraException::CANT_START_ACQUISITION);
	}

	ICamera* new_camera_device()
	{
		return new CameraPhotonFocus();
	}

	PvPixelType CameraPhotonFocus::get_pixel_type(std::string input)
	{
		// pixel_type not found in config file
		if (input == "")
			return pixel_type_;
		if (input == "Mono8")
			return PvPixelMono8;

		desc_.depth = 2.0f;
		if (input == "Mono10")
			return PvPixelMono10;
		if (input == "Mono12")
			return PvPixelMono12;
		throw CameraException(CameraException::CANT_SET_CONFIG);
	}

	void CameraPhotonFocus::display_image(PvImage *image)
	{
		std::cout << "------ Image information ------" << std::endl << std::endl;
		std::cout << "Width: " << image->GetWidth() << std::endl;
		std::cout << "Height: " << image->GetHeight() << std::endl;
		std::cout << "Bits: " << image->GetBitsPerPixel() << std::endl;
		std::cout << "Pixel type: " << image->GetPixelType() << std::endl;
		std::cout << "Pixel size: " << image->GetPixelSize(image->GetPixelType()) << std::endl;
		std::cout << "eff image size: " << image->GetEffectiveImageSize() << std::endl;
		std::cout << "image size: " << image->GetImageSize() << std::endl;
		std::cout << "required size: " << image->GetRequiredSize() << std::endl;
		std::cout << "offset x: " << image->GetOffsetX() << std::endl;
		std::cout << "offset y: " << image->GetOffsetY() << std::endl;
		std::cout << "padding x: " << image->GetPaddingX() << std::endl;
		std::cout << "padding y: " << image->GetPaddingY() << std::endl;
		std::cout << std::endl;
	}

	void CameraPhotonFocus::DumpGenParameterArray(PvGenParameterArray *aArray, std::string Param)
	{
		// Getting array size
		PvUInt32 lParameterArrayCount = aArray->GetCount();

		// Traverse through Array and print out parameters available
		for (PvUInt32 x = 0; x < lParameterArrayCount; x++)
		{
			// Get a parameter
			PvGenParameter *lGenParameter = aArray->Get(x);

			// Don't show invisible parameters - display everything up to Guru
			if (!lGenParameter->IsVisible(PvGenVisibilityGuru))
				continue;

			// Get and print parameter's name
			PvString lGenParameterName, lCategory;
			lGenParameter->GetCategory(lCategory);
			lGenParameter->GetName(lGenParameterName);
			if (Param != "" && lCategory.GetAscii() != Param)
				continue;
			std::cout << lCategory.GetAscii() << ":" << lGenParameterName.GetAscii() << ", ";

			// Parameter available?
			if (!lGenParameter->IsAvailable())
			{
				std::cout << "{Not Available}" << std::endl;
				continue;
			}

			// Parameter readable?
			if (!lGenParameter->IsReadable())
			{
				std::cout << "{Not readable}" << std::endl;
				continue;
			}

			// Get the parameter type
			PvGenType lType;
			lGenParameter->GetType(lType);
			switch (lType)
			{
				// If the parameter is of type PvGenTypeInteger
			case PvGenTypeInteger:
			{
				PvInt64 lValue;
				static_cast<PvGenInteger *>(lGenParameter)->GetValue(lValue);
				std::cout << "Integer: " << lValue;
			}
			break;
			// If the parameter is of type PvGenTypeEnum
			case PvGenTypeEnum:
			{
				PvString lValue;
				static_cast<PvGenEnum *>(lGenParameter)->GetValue(lValue);
				std::cout << "Enum: " << lValue.GetAscii();
			}
			break;
			// If the parameter is of type PvGenTypeBoolean
			case PvGenTypeBoolean:
			{
				bool lValue;
				static_cast<PvGenBoolean *>(lGenParameter)->GetValue(lValue);
				if (lValue)
				{
					std::cout << "Boolean: TRUE";
				}
				else
				{
					std::cout << "Boolean: FALSE";
				}
			}
			break;
			// If the parameter is of type PvGenTypeString
			case PvGenTypeString:
			{
				PvString lValue;
				static_cast<PvGenString *>(lGenParameter)->GetValue(lValue);
				std::cout << "String: " << lValue.GetAscii();
			}
			break;
			// If the parameter is of type PvGenTypeCommand
			case PvGenTypeCommand:
				std::cout << "Command";
				break;
				// If the parameter is of type PvGenTypeFloat
			case PvGenTypeFloat:
			{
				double lValue;
				static_cast<PvGenFloat *>(lGenParameter)->GetValue(lValue);
				std::cout << "Float: " << lValue;
			}
			break;
			}
			std::cout << std::endl;
		}
	}
}