#ifndef CAMERA_IXON_HH
# define CAMERA_IXON_HH

# include "camera.hh"
#include "atmcd32d.h"  

namespace camera
{
	class CameraIxon : public Camera
	{
		/*! Number of buffer. Each buffer contains one frame. */
		static const unsigned int NBUFFERS = 10;
	public:
		CameraIxon();
		~CameraIxon();

		virtual void init_camera() override;
		virtual void start_acquisition() override;
		virtual void stop_acquisition() override;
		virtual void shutdown_camera() override;
		virtual void* get_frame() override;

	private:
		virtual void load_default_params() override;
		virtual void load_ini_params() override;
		virtual void bind_params() override;

		//void create_buffers();
		//void delete_buffers();

	private:
		long device_handle;
		unsigned short* image_;
		int trigger_mode_;
		float shutter_close_;
		float shutter_open_;
		int ttl_;
		int shutter_mode_;
		//unsigned char* buffers_[NBUFFERS];
	};
}
#endif /* !CAMERA_ZYLA_HH */