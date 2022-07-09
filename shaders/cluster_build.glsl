#version 450
#pragma shader_stage(compute)

struct ClusterAabb {
	vec3 vs_min;
	vec3 vs_max;
};

layout (std140, set = 0, binding = 0) buffer General {
	mat4 perspective_inverse;
	uvec2 screen_extent;
	uvec4 cluster_grid;
	float znear;
	float zfar;
} general;

layout (std140, set = 0, binding = 1) buffer AabbBuffer {
	ClusterAabb aabbs[];
} aabb_buffer;

vec4 clip_to_view(vec4 clip) {
	vec4 view = general.perspective_inverse * clip;
	return view / view.w;
}

vec4 screen_to_clip(vec4 screen) {
	vec2 coords = screen.xy / general.screen_extent.xy;
	return vec4(vec2(coords.x, coords.y) * 2.0 - 1.0, screen.z, screen.w);
}

vec3 ray_intersect(vec3 eye, vec3 to, float z_dist) {
	vec3 normal = vec3(0.0, 0.0, 1.0);
	vec3 line = to - eye;

	float t = (z_dist - dot(normal, eye)) / dot(normal, line);

	return eye + line * t;
}

void main() {
	vec3 eye = vec3(0.0);

	uint tile_size = general.cluster_grid[3];
	uint tile_index = gl_WorkGroupID.x
		+ gl_WorkGroupID.y * gl_NumWorkGroups.x
		+ gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y;

	vec4 ss_min = vec4(gl_WorkGroupID.xy * tile_size, -1.0, 1.0);
	vec4 ss_max = vec4(vec2(gl_WorkGroupID.x + 1, gl_WorkGroupID.y + 1) * tile_size, -1.0, 1.0);

	vec3 vs_min = screen_to_clip(clip_to_view(ss_min)).xyz;
	vec3 vs_max = screen_to_clip(clip_to_view(ss_max)).xyz;

	float zratio = general.zfar / general.znear;
	float tile_near = -general.znear * pow(zratio, gl_WorkGroupID.z / float(gl_NumWorkGroups.z));
	float tile_far = -general.znear * pow(zratio, (gl_WorkGroupID.z + 1) / float(gl_NumWorkGroups.z));

	vec3 min_point_near = ray_intersect(eye, vs_min, tile_near);
	vec3 max_point_near = ray_intersect(eye, vs_max, tile_near);

	vec3 min_point_far = ray_intersect(eye, vs_min, tile_far);
	vec3 max_point_far = ray_intersect(eye, vs_max, tile_far);

	aabb_buffer.aabbs[tile_index] = ClusterAabb(
		min(min(min_point_near, min_point_far), min(max_point_near, max_point_far)),
		max(max(min_point_near, min_point_far), max(max_point_near, max_point_far))
	);
}
