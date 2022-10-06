#version 450

in vec2 texCoord;
uniform sampler2D tex;
uniform uint bitshift; 

out vec4 outColor;

void main() { outColor = texture(tex, texCoord) * (1 << bitshift); }
