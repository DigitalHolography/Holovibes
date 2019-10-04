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

int cross_thickness = 2;
int cross_length = 20;
int w_2 = window_width / 2;
int h_2 = window_height / 2;

void main()
{
	outColor = texture(tex, texCoord);
	if (display_cross)
	{
		vec4 pos = gl_FragCoord;
		if (w_2 - cross_thickness <= pos.x && pos.x <= w_2 + cross_thickness)
		{
			if (h_2 - cross_length <= pos.y && pos.y <= h_2 + cross_length)
			{
				outColor = vec4(1, 0, 0, 1);
			}
		}
		else if (h_2 - cross_thickness <= pos.y && pos.y <= h_2 + cross_thickness)
		{
			if (w_2 - cross_length <= pos.x && pos.x <= w_2 + cross_length)
			{
				outColor = vec4(1, 0, 0, 1);
			}
		}
	}
}
