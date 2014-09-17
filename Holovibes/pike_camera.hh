#ifndef PIKE_CAMERA_HH
# define PIKE_CAMERA_HH

# include <string>
# include "camera.hh"

namespace cam_driver
{

	class PikeCamera : public cam_driver::Camera
	{
	public:
		PikeCamera(std::string name) : Camera(name)
		{
		};

		~PikeCamera()
		{
		};

		virtual bool init_camera();
		virtual void start_acquisition();
		virtual void stop_acquisition();
		virtual void shutdown_camera();
	};
}

#endif