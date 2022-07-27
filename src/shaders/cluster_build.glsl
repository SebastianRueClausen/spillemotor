#version 450
#pragma shader_stage(compute)

#include "cluster_general.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

readonly layout (std140, set = 0, binding = 0) uniform Cluster {
	ClusterInfo cluster_info;
};

layout (std140, set = 0, binding = 1) uniform Proj {
	mat4 proj;
	mat4 inverse_proj;
	vec2 screen_dimensions;
	vec2 z_plane;
};

layout (std430, set = 0, binding = 2) buffer Set1 {
	Aabb aabbs[];
};

vec4 screen_to_view(vec2 screen, float z) {
	vec2 coords = screen / screen_dimensions.xy;
	vec4 clip = vec4(vec2(coords.x, 1.0 - coords.y) * 2.0 - 1.0, z, 1);
	vec4 view = inverse_proj * clip;
	return view / view.w;
}

uint cluster_index(uvec3 coords) {
	return coords.z * cluster_info.subdivisions.x * cluster_info.subdivisions.y
		+ coords.y * cluster_info.subdivisions.x
		+ coords.x;
}

void main() {
	uvec3 cluster_coords = gl_WorkGroupID;
	uint cluster_index = cluster_index(cluster_coords);

	vec2 screen_min = vec2(cluster_coords.xy * cluster_info.cluster_size.xy);
	vec2 screen_max = vec2((cluster_coords.xy + 1.0) * cluster_info.cluster_size.xy);

	vec3 view_min = screen_to_view(screen_min, 1.0).xyz;
	vec3 view_max = screen_to_view(screen_max, 1.0).xyz;

	view_min.y = -view_min.y;
	view_max.y = -view_max.y;

	float z_near = z_plane.x;
	float z_far = z_plane.y;

	float z_far_over_z_near = z_far / z_near;

	float view_near = -z_near * pow(
		z_far_over_z_near,
		cluster_coords.z / float(cluster_info.subdivisions.z)
	);

	float view_far = -z_near * pow(
		z_far_over_z_near,
		(cluster_coords.z + 1) / float(cluster_info.subdivisions.z)
	);

	vec3 min_near = view_min * view_near / view_min.z;
	vec3 max_near = view_max * view_near / view_max.z;

	vec3 min_far = view_min * view_far / view_min.z;
	vec3 max_far = view_max * view_far / view_max.z;

	aabbs[cluster_index] = Aabb(
		vec4(min(min_near, min(max_near, min(min_far, max_far))), 1.0),
		vec4(max(min_near, max(max_near, max(min_far, max_far))), 1.0)
	);
}
