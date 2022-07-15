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

vec3 ray_intersect_plane(vec3 start, vec3 end, Plane plane) {
	vec3 ray = end - start;
	float t = (plane.dist - dot(plane.normal, start)) / dot(plane.normal, ray);
	return start + t * ray;
}

uint cluster_index(uvec3 coords) {
	return coords.x + cluster_info.subdivisions.x * (coords.y + cluster_info.subdivisions.y * coords.z);
}

void main() {
	uvec3 cluster_coords = gl_WorkGroupID;
	uint cluster_index = cluster_index(cluster_coords);

	vec2 screen_min = vec2(cluster_coords.xy * cluster_info.cluster_size);
	vec2 screen_max = vec2((cluster_coords.xy + 1.0) * cluster_info.cluster_size);

	vec4 view_min = screen_to_view(screen_min, 1.0);
	vec4 view_max = screen_to_view(screen_max, 1.0);

	float cluster_near = -cluster_info.z_near * pow(abs(cluster_info.k_near), cluster_coords.z);
	float cluster_far = -cluster_info.z_near * pow(abs(cluster_info.k_near), cluster_coords.z + 1);

	vec3 normal = vec3(0.0, 0.0, 1.0);

	Plane near_plane = Plane(normal, cluster_near);
    Plane far_plane = Plane(normal, cluster_far);

	vec3 eye = vec3(0.0, 0.0, 0.0);

	vec3 near_min = ray_intersect_plane(eye, vec3(view_min), near_plane);
	vec3 near_max = ray_intersect_plane(eye, vec3(view_max), near_plane);

	vec3 far_min = ray_intersect_plane(eye, vec3(view_min), far_plane);
	vec3 far_max = ray_intersect_plane(eye, vec3(view_max), far_plane);

	aabbs[cluster_index] = Aabb(
		vec4(min(near_min, min(near_max, min(far_min, far_max))), 1.0),
		vec4(max(near_min, max(near_max, max(far_min, far_max))), 1.0)
	);
}
