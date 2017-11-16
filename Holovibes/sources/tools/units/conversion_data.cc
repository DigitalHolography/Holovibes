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

#include <cassert>

#include "units\conversion_data.hh"
#include "units\unit.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes
{
	using gui::BasicOpenGLWindow;
	using gui::SliceYZ;
	using gui::SliceXZ;
	namespace units
	{
		ConversionData::ConversionData(const BasicOpenGLWindow& window)
			: window_(&window)
		{}

		ConversionData::ConversionData(const BasicOpenGLWindow* window)
			: window_(window)
		{}

		float ConversionData::window_size_to_opengl(int val, Axis axis) const
		{
			assert(window_);
			if (axis == Axis::HORIZONTAL && window_->width() > window_->height()
				&& window_->getKindOfView() != SliceXZ && window_->getKindOfView() != SliceYZ)
			{
				double multiplier = static_cast<double>(window_->width()) / static_cast<double>(window_->height());
				val *= multiplier;
			}
			auto res = (static_cast<float>(val) * 2.f / static_cast<float>(get_window_size(axis))) - 1.f;
			return axis == Axis::VERTICAL ? -res : res;
		}

		float ConversionData::fd_to_opengl(int val, Axis axis) const
		{
			assert(window_);
			auto res = (static_cast<float>(val) * 2.f / static_cast<float>(get_fd_size(axis))) - 1.f;
			return axis == Axis::VERTICAL ? -res : res;
		}

int ConversionData::opengl_to_window_size(float val, Axis axis) const
{
	assert(window_);
	if (axis == Axis::VERTICAL)
		val *= -1;
	int res = ((val + 1.f) / 2.f) * get_window_size(axis);
	if (window_->width() > window_->height() && axis == Axis::HORIZONTAL
		&& window_->getKindOfView() != SliceXZ && window_->getKindOfView() != SliceYZ)
	{
		double divider = static_cast<double>(window_->width()) / static_cast<double>(window_->height());
		res /= divider;
	}
	return res;
}

		int ConversionData::opengl_to_fd(float val, Axis axis) const
		{
			assert(window_);
			if (axis == Axis::VERTICAL)
				val *= -1;
			return ((val + 1.f) / 2.f) * get_fd_size(axis);
		}

void ConversionData::transform_from_fd(float& x, float& y) const
{
	glm::vec3 input{ x, y, 1.0f };
	const auto& matrix = window_->getTransformMatrix();
	auto output = matrix * input;
	x = output[0];
	y = output[1];
}

void ConversionData::transform_to_fd(float& x, float& y) const
{
	glm::vec3 input{ x, y, 1 };
	auto matrix = window_->getTransformInverseMatrix();
	auto output = matrix * input;
	x = output[0];
	y = output[1];
}
int ConversionData::get_window_size(Axis axis) const
{
	switch (axis)
	{
	case Axis::HORIZONTAL:
		return window_->width();
	case Axis::VERTICAL:
		return window_->height();
	default:
		throw std::exception("Unreachable code");
	}
}

		int ConversionData::get_fd_size(Axis axis) const
		{
			switch (axis)
			{
			case Axis::HORIZONTAL:
				return window_->getFd().width;
			case Axis::VERTICAL:
				return window_->getFd().height;
			default:
				throw std::exception("Unreachable code");
			}
		}
	}
}
