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
#include <cmath>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#include "camera_phantom.hh"

namespace camera
{
	CameraPhantom::CameraPhantom()
		: Camera("phantom.ini")
	{
		gentl_ = std::make_unique<Euresys::EGenTL>();
		grabber_ = std::make_unique<EuresysCustomGrabber>(*gentl_);

		name_ = "Phantom S710";
		pixel_size_ = 12;

		fd_.width = 1280;
		fd_.height = 200;
		fd_.depth = 1;
		fd_.byteEndian = Endianness::BigEndian;

		load_default_params();
		if (ini_file_is_open())
		{
			load_ini_params();
			ini_file_.close();
		}

		init_camera();
	}

	void CameraPhantom::init_camera()
	{
		bind_params();

		grabber_->enableEvent<Euresys::NewBufferData>();

		buffers_ = std::vector<unsigned char *>(nb_buffers_, nullptr);

		size_t size = grabber_->getWidth() * grabber_->getHeight();
		for (size_t i = 0; i < nb_buffers_; ++i)
		{
			unsigned char *ptr, *devicePtr;

			cudaError_t alloc_res = cudaHostAlloc(&ptr, size, cudaHostAllocMapped);
			cudaError_t device_ptr_res = cudaHostGetDevicePointer(&devicePtr, ptr, 0);
			if (alloc_res != cudaSuccess || device_ptr_res != cudaSuccess)
				std::cerr << "[CAMERA] Could not allocate buffers." << std::endl;

			buffers_[i] = ptr;
			grabber_->announceAndQueue(Euresys::UserMemory(ptr, size, devicePtr));
		}
	}

	void CameraPhantom::start_acquisition()
	{
		grabber_->start();
	}

	void CameraPhantom::stop_acquisition()
	{
		for (size_t i = 0; i < nb_buffers_; ++i)
			cudaFreeHost(buffers_[i]);

		grabber_->stop();
	}

	void CameraPhantom::shutdown_camera()
	{
		return;
	}

	CapturedFramesDescriptor CameraPhantom::get_frames()
	{
		grabber_->processEvent<Euresys::NewBufferData>(FRAME_TIMEOUT);

		return CapturedFramesDescriptor(grabber_->last_device_ptr_, 1, true);
	}

	void CameraPhantom::load_default_params()
	{
		nb_buffers_ = 64;
		width_ = 1280;
		height_ = 200;
		roi_x_ = 0;
		roi_y_ = 0;
		frame_period_ = 1000000 / 3000; // 1e+6 / FPS
		exposure_time_ = 5000;
	}

	void CameraPhantom::load_ini_params()
	{
		const boost::property_tree::ptree& pt = get_ini_pt();
		nb_buffers_ = pt.get<unsigned int>("phantom.nb_buffers", nb_buffers_);
		width_ = pt.get<unsigned int>("phantom.width", width_);
		height_ = pt.get<unsigned int>("phantom.height", height_);
		roi_x_ = pt.get<unsigned int>("phantom.roi_x", roi_x_);
		roi_y_ = pt.get<unsigned int>("phantom.roi_x", roi_y_);
		frame_period_ = pt.get<float>("phantom.frame_period", frame_period_);
		exposure_time_ = pt.get<float>("phantom.exposure_time", exposure_time_);
	}

	void CameraPhantom::bind_params()
	{
		// Camera configuration
		grabber_->setInteger<Euresys::RemoteModule>("Width", width_);
		grabber_->setInteger<Euresys::RemoteModule>("Height", height_);
		// grabber_->setInteger<Euresys::RemoteModule>("OffsetX", roi_x_);
		// grabber_->setInteger<Euresys::RemoteModule>("OffsetY", roi_y_);

		// Frame grabber configuration
		// grabber_->setFloat<Euresys::DeviceModule>("CycleMinimumPeriod", frame_period_);
		// grabber_->setFloat<Euresys::DeviceModule>("ExposureTime", exposure_time_);
	}

	ICamera* new_camera_device()
	{
		return new CameraPhantom();
	}
}