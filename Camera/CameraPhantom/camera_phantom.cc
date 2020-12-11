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
		pixel_size_ = 20;

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
		fd_.width = grabber_->getWidth();
		fd_.height = grabber_->getHeight();
		fd_.depth = 1;
		fd_.byteEndian = Endianness::BigEndian;

		grabber_->enableEvent<Euresys::NewBufferData>();

		buffers_ = std::vector<unsigned char *>(nb_buffers_, nullptr);

		size_t size = fd_.width * fd_.height;
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
	}

	void CameraPhantom::load_ini_params()
	{
		const boost::property_tree::ptree& pt = get_ini_pt();
		nb_buffers_ = pt.get<unsigned int>("phantom.nb_buffers", nb_buffers_);
	}

	void CameraPhantom::bind_params()
	{
		return;
	}

	ICamera* new_camera_device()
	{
		return new CameraPhantom();
	}
}