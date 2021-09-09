#version 450

in float passColor;
out vec4 outColor;

void main() { outColor = vec4(passColor / 65535.f); }
