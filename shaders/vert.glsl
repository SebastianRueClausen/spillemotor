#version 450
#pragma shader_stage(vertex)
// #extension GL_EXT_debug_printf:enable

layout (set = 0, binding = 0) uniform UniformBufferObject {
	mat4 perspective;
	mat4 view;
} ubo;

layout (push_constant) uniform PushConstant {
	mat4 transform;
} pc;

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_texcoord;

layout (location = 0) out vec2 out_texcoord;

void main() {
	out_texcoord = in_texcoord;

	gl_Position = ubo.perspective * ubo.view * pc.transform * vec4(in_position, 1.0);
}
