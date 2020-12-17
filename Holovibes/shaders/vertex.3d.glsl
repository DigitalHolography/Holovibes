/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#version 450

layout(location = 0) in vec3	vertex;
layout(location = 1) in float	color;

uniform mat4 MVP;

out float passColor;

void main()
{
	passColor = color;
	gl_Position = MVP * vec4(vertex, 1.f);
}