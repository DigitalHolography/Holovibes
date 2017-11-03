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

#include "units\conversion_data.hh"

using holovibes::units::ConversionData;


ConversionData::ConversionData(const int& window_size, const int& fd_size)
	: window_size_(window_size)
	, fd_size_(fd_size)
{}

float ConversionData::window_size_to_opengl(int val) const
{
	// FIXME
	return val;
}

float ConversionData::fd_to_opengl(int val) const
{
	// FIXME
	return val;
}

int ConversionData::opengl_to_window_size(float val) const
{
	// FIXME
	return val;
}

int ConversionData::opengl_to_fd(float val) const
{
	// FIXME
	return val;
}
