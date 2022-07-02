#version 450
#pragma shader_stage(vertex)

layout (set = 0, binding = 0) uniform UniformBufferObject {
	mat4 perspective;
	mat4 view;
} ubo;

layout (push_constant) uniform PushConstant {
	mat4 transform;
} pc;

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;

layout (location = 0) out vec2 out_texcoord;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec3 out_ws_pos;

void main() {
	vec4 pos = vec4(in_pos, 1.0);

	out_texcoord = in_texcoord;
	out_normal = normalize(mat3(pc.transform) * in_normal);
	out_ws_pos = vec3(pc.transform * pos);

	gl_Position = ubo.perspective * ubo.view * pc.transform * pos;
}
