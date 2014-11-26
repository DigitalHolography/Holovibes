//#include "stdafx.h"
#include "camera_ixon.hh"
#include "camera_exception.hh"

namespace camera
{
	CameraIxon::CameraIxon()
		: Camera("Ixon.ini")
	{
		name_ = "ixon";
		long nb_cam;
		load_default_params();
		GetAvailableCameras(&nb_cam);
		if (nb_cam < 1)
			throw CameraException(name_, CameraException::NOT_CONNECTED);
		
		std::cout << nb_cam << std::endl;//remove me
		
		//if (ini_file_is_open())
		//	load_ini_params();
	}

	CameraIxon::~CameraIxon()
	{

	}

	void CameraIxon::init_camera()
	{
		GetCameraHandle(0, &device_handle);
		if (SetCurrentCamera(device_handle) == DRV_P1INVALID)
			throw CameraException(name_, CameraException::NOT_INITIALIZED);
		char aBuffer[256];
		GetCurrentDirectory(256, aBuffer);
		if (Initialize(aBuffer) != DRV_SUCCESS)
			throw CameraException(name_, CameraException::NOT_INITIALIZED);
		int x, y;
		GetDetector(&x, &y);
		std::cout << x << "    " << y << std::endl;
		image_ = (unsigned short*)malloc(desc_.frame_size());
	}

	void CameraIxon::start_acquisition()
	{
		unsigned int error;
		error = SetAcquisitionMode(5); // RUN TILL ABORT
		if (error != DRV_SUCCESS)
			throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
		error = SetReadMode(4);
		if (error != DRV_SUCCESS)
			throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
		error = SetExposureTime(exposure_time_);
		if (error != DRV_SUCCESS)
			throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
		error = SetShutter(ttl_, shutter_mode_, shutter_open_, shutter_close_);
		if (error != DRV_SUCCESS)
			throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
		error = SetTriggerMode(trigger_mode_);
		if (error != DRV_SUCCESS)
			throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
		//SetReadMode(4);
		error = SetImage(1, 1, 1, desc_.width, 1, desc_.height);
		if (error != DRV_SUCCESS)
			throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
		//SetFrameTransferMode()//ne effcet in kinetics or single scan
		//SetAcquisitionMode(5); // 1 single scan / 2 accumulates / 3 kinectics / 4 fast kin / 5 run till abort
		long nbuff;
		if (GetSizeOfCircularBuffer(&nbuff) == DRV_NOT_INITIALIZED)
			std::cout << "drv no init" << std::endl;
		std::cout << "nbbuff " << nbuff << std::endl;
		//SetCoolerMode(0);
		StartAcquisition();
		
	}

	void CameraIxon::stop_acquisition()
	{	
	}

	void CameraIxon::shutdown_camera()
	{
		ShutDown();
	}

	void* CameraIxon::get_frame()
	{
		long first;
		long last;
		GetNumberNewImages(&first, &last);
		std::cout << "first: " << first << " last: " << last << std::endl;
		SendSoftwareTrigger();
		WaitForAcquisition();
		unsigned int error = GetNewData16(image_, desc_.width * desc_.height);

		if (error != DRV_SUCCESS && error != DRV_NO_NEW_DATA)
			throw CameraException(name_, CameraException::CANT_GET_FRAME);
		else
		return ((void*)image_);
	}

	void CameraIxon::load_default_params()
	{
		desc_.width = 1002;
		desc_.height = 1002;
		desc_.depth = 2;
		desc_.pixel_size = 7.4f;
		desc_.endianness = BIG_ENDIAN;
		exposure_time_ = 0.1;
		trigger_mode_ = 10; //0
		shutter_close_ = 0;
		shutter_open_ = 0;
		ttl_ = 1;
		shutter_mode_ = 5; //0
	}

	void CameraIxon::load_ini_params()
	{
		/* Use the default value in case of fail. */
		const boost::property_tree::ptree& pt = get_ini_pt();
		desc_.width = pt.get<unsigned short>("ixon.sensor_width", desc_.width);
		desc_.height = pt.get<unsigned short>("ixon.sensor_height", desc_.height);

		exposure_time_ = pt.get<float>("ixon.exposure_time", exposure_time_);
		trigger_mode_ = pt.get<int>("ixon.trigger_mode", trigger_mode_);
		shutter_close_ = pt.get<float>("ixon.shutter_close", shutter_close_);
		shutter_open_ = pt.get<float>("ixon.shutter_open", shutter_close_);
		ttl_ = pt.get<int>("ixon.ttl", ttl_);
		shutter_mode_ = pt.get<int>("ixon.shutter_mode", shutter_mode_);
	}

	void CameraIxon::bind_params()
	{
		
	}

	

	
}