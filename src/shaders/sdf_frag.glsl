#version 450
#pragma shader_stage(fragment)

layout (set = 0, binding = 0) uniform sampler2D color_sampler;

layout (location = 0) in vec2 in_texcoord;
layout (location = 0) out vec4 out_color;

void main() {
	float dist = texture(color_sampler, in_texcoord).r;
	float width = fwidth(dist);
	float alpha = smoothstep(0.5 - width, 0.5 + width, dist);
	out_color = vec4(alpha);
}
