
const uint MAX_LIGHT_COUNT = 256;

struct ClusterInfo {
	uvec4 subdivisions;
	uvec2 cluster_size;
	vec2 depth_factors;
};

struct Aabb {
	vec4 min_point;
	vec4 max_point;
};

struct Plane {
	vec3 normal;
	float dist;
};

struct Sphere {
	vec3 pos;
	float radius;
};

struct PointLight {
	vec4 pos;
	vec3 lum;
	float radius;
};

struct DirLight {
	vec4 dir;
	vec4 irradiance;
};

struct LightPos {
	vec3 view_pos;
	float radius;
};

const uint LIGHT_MASK_WORD_COUNT = uint(ceil(float(MAX_LIGHT_COUNT)) / 32.0);

struct LightMask {
	uint mask[LIGHT_MASK_WORD_COUNT];
};

uint cluster_index(uvec3 subdivisions, uvec3 coords) {
	return coords.z * subdivisions.x * subdivisions.y
		+ coords.y * subdivisions.x
		+ coords.x;
}

