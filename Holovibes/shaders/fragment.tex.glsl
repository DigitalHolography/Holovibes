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

#version 450

in		vec2		texCoord;
uniform sampler2D	tex;

out		vec4		outColor;

uniform bool display_cross;
uniform int window_width;
uniform int window_height;

int cross_thickness = 1;
int cross_length = 20;
int w_2 = window_width / 2;
int h_2 = window_height / 2;
int w_4 = window_width / 4;
int h_4 = window_height / 4;

bool val_between(float val, float lo, float hi)
{
	return lo <= val && val <= hi;
}

bool pos_in_cross(vec2 pos)
{
	if (val_between(pos.x, w_2 - cross_thickness, w_2 + cross_thickness)
		&& val_between(pos.y, h_2 - cross_length, h_2 + cross_length))
	{
		return true;
	}
	if (val_between(pos.y, h_2 - cross_thickness, h_2 + cross_thickness)
		&& val_between(pos.x, w_2 - cross_length, w_2 + cross_length))
	{
		return true;
	}
	return false;
}

bool pos_in_border(vec2 pos)
{
	if (val_between(pos.x, w_4 - cross_thickness, w_4 * 3 + cross_thickness))
	{
		if (val_between(pos.y, h_4 - cross_thickness, h_4 + cross_thickness)
			|| val_between(pos.y, h_4 * 3 - cross_thickness, h_4 * 3 + cross_thickness))
		{
			return true;
		}
	}
	if (val_between(pos.y, h_4 - cross_thickness, h_4 * 3 + cross_thickness))
	{
		if (val_between(pos.x, w_4 - cross_thickness, w_4 + cross_thickness)
			|| val_between(pos.x, w_4 * 3 - cross_thickness, w_4 * 3 + cross_thickness))
		{
			return true;
		}
	}
	return false;
}

void main()
{
	outColor = texture(tex, texCoord);
	if (display_cross)
	{
		vec4 pos = gl_FragCoord;
		if (pos_in_cross(pos.xy) || pos_in_border(pos.xy))
		{
			outColor = vec4(1.0, 0.0, 0.0, 0.5);
		}
	}
}
