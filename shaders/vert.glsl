#version 450
#pragma shader_stage(vertex)

layout (set = 0, binding = 0) uniform View {
	vec4 eye;
	mat4 view;
	mat4 proj_view;
};

layout (push_constant) uniform PushConstant {
	mat4 transform;
	mat4 inverse_transpose_transform;
};

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;
layout (location = 3) in vec4 in_tangent;

layout (location = 0) out vec2 out_texcoord;

layout (location = 1) out vec3 out_world_normal;
layout (location = 2) out vec4 out_world_tangent;
layout (location = 3) out vec3 out_world_bitangent;
layout (location = 4) out vec4 out_world_position;

layout (location = 5) out float out_view_z;

void main() {
	vec4 position = vec4(in_position, 1.0);
	vec4 world = transform * position;

	out_texcoord = in_texcoord;

	out_world_normal = normalize(mat3(inverse_transpose_transform) * in_normal);
	out_world_tangent = normalize(vec4(mat3(transform) * in_tangent.xyz, in_tangent.w));
	out_world_bitangent = out_world_tangent.w * cross(out_world_tangent.xyz, out_world_normal);
	out_world_position = world;

	out_view_z = (view * world).z;

	gl_Position = proj_view * world;
}
