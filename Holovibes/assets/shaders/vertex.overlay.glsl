#version 450

layout(location = 2) in vec2 vertex;
layout(location = 3) in vec3 color;

uniform vec2 translation;
uniform vec2 scale;
uniform float alpha;

out vec4 passColor;

void main()
{
    passColor = vec4(color, alpha);
    gl_Position = vec4(translation + vertex * scale, 0.f, 1.f);
}
