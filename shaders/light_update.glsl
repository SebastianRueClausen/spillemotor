#version 450
#pragma shader_stage(compute)

#include "cluster_general.glsl"

const uint THREAD_COUNT = 64;

layout (local_size_x = THREAD_COUNT, local_size_y = 1, local_size_z = 1) in;

readonly layout (std140, set = 0, binding = 0) uniform View {
	vec4 eye;
	mat4 view;
	mat4 inverse_view;
	mat4 proj_view;
};

readonly layout (std430, set = 0, binding = 1) buffer Lights {
	uint point_light_count;

	DirLight dir_light;
	PointLight point_lights[];
};

writeonly layout (std430, set = 0, binding = 2) buffer LightPositions {
	vec4 light_positions[];
};

void main() {
	uint light_index = gl_LocalInvocationIndex + THREAD_COUNT * gl_WorkGroupID.x;
	if (light_index < point_light_count) {
		light_positions[light_index] = view * point_lights[light_index].pos;
	}
}
