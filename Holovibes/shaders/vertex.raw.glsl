/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#version 450

layout(location = 0) in vec2	xy;
layout(location = 1) in vec2	uv;

uniform vec2	translate;
uniform mat4	mvp;

out vec2	texCoord;

void main()
{
	texCoord = uv + translate;
	gl_Position = vec4((mvp * vec4(xy, 0.f, 1.f)).xy, 0.f, 1.f);
}