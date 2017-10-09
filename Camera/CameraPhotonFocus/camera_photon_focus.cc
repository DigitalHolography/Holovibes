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

#include "camera_photon_focus.hh"

namespace camera
{
	CameraPhotonFocus::CameraPhotonFocus()
		: Camera("adimec.ini")
	{
	}

	void CameraPhotonFocus::init_camera()
	{
	}

	void CameraPhotonFocus::start_acquisition()
	{
	}

	void CameraPhotonFocus::stop_acquisition()
	{
	}

	void CameraPhotonFocus::shutdown_camera()
	{
	}

	void* CameraPhotonFocus::get_frame()
	{
		return nullptr;
	}

	void CameraPhotonFocus::load_default_params()
	{
	}

	void CameraPhotonFocus::load_ini_params()
	{
	}

	void CameraPhotonFocus::bind_params()
	{
	}

	ICamera* new_camera_device()
	{
		return new CameraPhotonFocus();
	}
}