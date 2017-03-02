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

layout(location = 0) in vec2	xy;
layout(location = 1) in vec2	uv;
uniform float	angle;
uniform int		flip;

out vec2	texCoord;

mat2 rotate2d(float agl){
    return mat2(cos(agl),-sin(agl),
                sin(agl),cos(agl));
}

void main()
{
	texCoord = (flip == 1) ? vec2(uv.x, 1.f - uv.y) : uv;
    gl_Position = vec4(xy * rotate2d(angle), 0.0f, 1.0f);
}
