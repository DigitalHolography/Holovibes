/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#version 450

layout(location = 2) in vec2	vertex;
layout(location = 3) in vec3	color;

uniform float alpha;

out vec4 passColor;

void main()
{
    passColor = vec4(color, alpha);
    gl_Position = vec4(vertex, 0.f, 1.f);
}
