#version 450
#pragma shader_stage(vertex)

layout (push_constant) uniform PushConstant {
	mat4 proj_transform;
};

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_texcoord;

layout (location = 0) out vec2 out_texcoord;

void main() {
	out_texcoord = in_texcoord;
	gl_Position = proj_transform * vec4(in_position, 1.0);
}
