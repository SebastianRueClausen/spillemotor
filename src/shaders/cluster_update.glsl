#version 450
#pragma shader_stage(compute)

#include "cluster_general.glsl"

const uint THREADS_PER_CLUSTER = 64;

layout(local_size_x = THREADS_PER_CLUSTER, local_size_y = 1, local_size_z = 1) in;

readonly layout (std430, set = 0, binding = 0) buffer Lights {
	uint point_light_count;

	DirLight dir_light;
	PointLight point_lights[];
};

readonly layout (std430, set = 0, binding = 1) buffer LightPositions {
	LightPos light_positions[];
};

readonly layout (std430, set = 0, binding = 2) buffer Aabbs {
	Aabb aabbs[];
};

writeonly layout (std430, set = 0, binding = 3) buffer LightMasks {
	LightMask light_masks[];
};

shared LightMask shared_mask;

bool sphere_intersects_aabb(Aabb aabb, Sphere sphere) {
	vec3 closest = max(vec3(aabb.min_point), min(sphere.pos, vec3(aabb.max_point)));
	vec3 dist = closest - sphere.pos;
	return dot(dist, dist) <= sphere.radius * sphere.radius;
}

void main() {
	if (gl_LocalInvocationID.x == 0) {
		for (uint i = 0; i < LIGHT_MASK_WORD_COUNT; ++i) {
			shared_mask.mask[i] = 0;
		}
	}

	memoryBarrierShared();

	uint cluster_index = cluster_index(gl_NumWorkGroups, gl_WorkGroupID);
	Aabb aabb = aabbs[cluster_index];

	uint start = gl_LocalInvocationID.x;

	for (uint i = start; i < point_light_count; i += THREADS_PER_CLUSTER) {
		LightPos light = light_positions[i];

		Sphere sphere = Sphere(light.view_pos, light.radius);

		if (sphere_intersects_aabb(aabb, sphere)) {
			uint word = i / 32;
			uint bit = i % 32;

			atomicOr(shared_mask.mask[word], 1 << bit);
		}
	}

	memoryBarrierShared();
	barrier();

	if (gl_LocalInvocationID.x == 0) {
		light_masks[cluster_index] = shared_mask;	
	}
}
