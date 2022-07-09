#version 450
#pragma shader_stage(compute)

struct ClusterAabb {
	vec3 vs_min;
	vec3 vs_max;
};

struct PointLight {
	vec3 pos;
	vec3 lum;
	float radius;
};

readonly layout (std140, set = 1, binding = 0) buffer LightGeneral {
	mat4 perspective_inverse;
	uvec2 screen_extent;
	uvec4 cluster_grid;
	float znear;
	float zfar;
} light_general;

readonly layout (std140, set = 0, binding = 1) buffer LightBuffer {
	uint count;
	uint capacity;
	PointLight lights[];
} light_buffer;

readonly layout (std140, set = 0, binding = 2) buffer AabbBuffer {
	ClusterAabb aabbs[];
} aabb_buffer;


