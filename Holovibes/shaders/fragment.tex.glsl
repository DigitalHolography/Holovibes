#version 450

in vec2 texCoord;
uniform sampler2D tex;

out vec4 outColor;

void main() { outColor = texture(tex, texCoord); }
