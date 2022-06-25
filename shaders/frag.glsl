#version 450
#pragma shader_stage(fragment)

layout (set = 0, binding = 1) uniform sampler2D tex_sampler;
layout (location = 0) in vec2 in_texcoord;
layout (location = 0) out vec4 out_color;

void main() {
	vec4 color = texture(tex_sampler, in_texcoord);
	out_color = vec4(color.xyz, 1.0);
}
