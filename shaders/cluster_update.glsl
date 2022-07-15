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
	vec4 light_positions[];
};

readonly layout (std430, set = 0, binding = 2) buffer Aabbs {
	Aabb aabbs[];
};

writeonly layout (std430, set = 0, binding = 3) buffer LightIndices {
	uint light_indices[];
};

shared uint shared_light_count;
shared uint shared_light_indices[MAX_LIGHTS_IN_CLUSTER];

float square_dist_to_aabb(vec3 point, Aabb aabb) {
	float dist = 0.0;

	for (uint i = 0; i < 3; ++i) {
		float v = point[i];

		if (v < aabb.min_point[i]) {
			dist += pow(aabb.min_point[i] - v, 2);
		}

		if (v > aabb.max_point[i]) {
			dist += pow(v - aabb.max_point[i], 2);
		}
	}

	return dist;
}

bool aabb_intersect_sphere(Aabb aabb, Sphere sphere) {
	float square_dist = square_dist_to_aabb(sphere.pos, aabb);
	return square_dist <= sphere.radius * sphere.radius;
}

uint cluster_index(uvec3 coords) {
	return coords.x + gl_NumWorkGroups.x * (coords.y + gl_NumWorkGroups.y * coords.z);
}

void main() {
	if (gl_LocalInvocationID.x == 0) {
		shared_light_count = 0;
	}

	memoryBarrierShared();

	uint cluster_index = cluster_index(gl_WorkGroupID);
	Aabb aabb = aabbs[cluster_index];

	uint start = gl_LocalInvocationID.x;

	for (
		uint i = 0;
		i < point_light_count
			&& shared_light_count < MAX_LIGHTS_IN_CLUSTER
			&& start + i < point_light_count;
		i += THREADS_PER_CLUSTER
	) {
		uint light_index = start + i;

		PointLight light = point_lights[light_index];
		vec3 pos = vec3(light_positions[light_index]);

		Sphere sphere = Sphere(pos, light.radius);

		if (aabb_intersect_sphere(aabb, sphere)) {
			uint shared_index = atomicAdd(shared_light_count, 1);
			shared_light_indices[shared_index] = light_index;
		}
	}

	memoryBarrierShared();
	barrier();

	start = cluster_index * MAX_LIGHTS_IN_CLUSTER;

	uint light_count = shared_light_count;

	for (uint i = gl_LocalInvocationID.x; i < light_count; i += THREADS_PER_CLUSTER) {
		light_indices[start + i] = shared_light_indices[i];
	}

	if (gl_LocalInvocationID.x == 0) {
		if (light_count < MAX_LIGHTS_IN_CLUSTER) {
			light_indices[start + light_count] = LIGHT_INDEX_SENTINEL;
		}
	}
}
