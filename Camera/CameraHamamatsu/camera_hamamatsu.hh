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

/*! \file
*
* Camera Hamamatsu C11440 */
#pragma once

#include <camera.hh>
#include "camera_exception.hh"

//#pragma comment(lib,"dcamapi.lib")
#define WIN32
#include "dcamapi.h"
#include "dcamprop.h"

namespace camera
{
	class CameraHamamatsu : public Camera
	{
	public:
		CameraHamamatsu();

		virtual ~CameraHamamatsu()
		{}

		virtual void init_camera() override;
		virtual void start_acquisition() override;
		virtual void stop_acquisition() override;
		virtual void shutdown_camera() override;
		virtual void *get_frame() override;

		long copybits_bw16(WORD* dsttopleft, long dstrowbytes, const BYTE * lut, const WORD * srctopleft, long srcrowbytes, long srcox, long srcoy, long srcwidth, long srcheight);


	private:
		virtual void load_ini_params() override;
		virtual void load_default_params() override;
		virtual void bind_params() override;

		std::unique_ptr<unsigned short[]> output_frame_;

		HDCAM hdcam_;
		long srcox_, srcoy_;
		unsigned short binning_;
		bool ext_trig_;
		_DCAMPROPMODEVALUE trig_connector_;
		_DCAMPROPMODEVALUE trig_polarity_;
		_DCAMPROPMODEVALUE readoutspeed_;
	};
}
