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

/*! \file tools.hh
 *
 * Generic, widely usable functions. */
#pragma once

class Color
{
public:
	Color()
		:components_{ 0 }
	{}
	Color(double hue, double red, double blue)
	{
		/* reference image for continous rainbow
		 * https://cdn.instructables.com/FTI/O536/IIG9YHKA/FTIO536IIG9YHKA.MEDIUM.jpg
		*/
		double x = (hue - red) / (blue - red);
		if (x < 0.25) {
			components_[0] = 1;
			components_[1] = x / 0.25;
			components_[2] = 0;
		}
		else if (x < 0.5) {
			components_[0] = 1- (x - 0.25) / 0.25 / 0.;
			components_[1] = 1;
			components_[2] = 0;
		}
		else if (x < 0.75) {
			components_[0] = 0;
			components_[1] = 1;
			components_[2] = (x - 0.5) / 0.25;
		}
		else {
			components_[0] = 0;
			components_[1] = 1 - (x - 0.75) / 0.25;
			components_[2] = 1;
		}
	}
	Color& operator*(double value) {
		for (int i = 0; i < 3; i++)
			components_[i] *= value;
	}
	Color& operator+=(const Color& c) {
		for (int i = 0; i < 3; i++)
			components_[i] += c.components_[i];
	}
	double operator[](int index) const {
		return components_[index];
	}

private:
	double components_[3];
};