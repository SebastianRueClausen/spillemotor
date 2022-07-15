#version 450
#pragma shader_stage(vertex)

layout (set = 0, binding = 0) uniform View {
	vec4 eye;
	mat4 view;
	mat4 inverse_view;
	mat4 proj_view;
};

layout (push_constant) uniform PushConstant {
	mat4 transform;
};

layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;

layout (location = 0) out vec2 out_texcoord;
layout (location = 1) out vec3 out_normal;
layout (location = 2) out vec4 out_ws_pos;
layout (location = 3) out float out_z_view;

void main() {
	vec4 pos = vec4(in_pos, 1.0);
	vec4 world = transform * pos;

	out_texcoord = in_texcoord;
	out_normal = normalize(mat3(transform) * in_normal);
	out_ws_pos = world;
	out_z_view = (view * world).z;

	gl_Position = proj_view * world;
}
