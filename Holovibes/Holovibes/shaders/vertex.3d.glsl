#version 450

layout(location = 0) in vec3 vertex;
layout(location = 1) in float color;

uniform mat4 MVP;

out float passColor;

void main()
{
    passColor = color;
    gl_Position = MVP * vec4(vertex, 1.f);
}