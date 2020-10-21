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
			return res;
		}

		int ConversionData::opengl_to_fd(float val, Axis axis) const
		{
			assert(window_);
			if (axis == Axis::VERTICAL)
				val *= -1;
			return ((val + 1.f) / 2.f) * get_fd_size(axis);
		}

		double ConversionData::fd_to_real(int val, Axis axis) const
		{
			assert(window_);
			auto cd = window_->getCd();
			auto fd = window_->getFd();
			float pix_size;
			if (window_->getKindOfView() == Hologram)
				pix_size = (cd->lambda * cd->zdistance) / (fd.width * cd->pixel_size * 1e-6);
			else if (window_->getKindOfView() == SliceXZ && axis == Axis::HORIZONTAL) {
				pix_size = (cd->lambda * cd->zdistance) / (fd.width * cd->pixel_size * 1e-6);
			}
			else if (window_->getKindOfView() == SliceYZ && axis == Axis::VERTICAL) {
				pix_size = (cd->lambda * cd->zdistance) / (fd.height * cd->pixel_size * 1e-6);
			}
			else
			{
				pix_size = std::pow(cd->lambda, 2) / 50E-9; // 50nm is an arbitrary value
			}

			return val * pix_size;
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
