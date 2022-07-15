
const uint MAX_LIGHT_COUNT = 128;
const uint MAX_LIGHTS_IN_CLUSTER = 32;
const uint LIGHT_INDEX_SENTINEL = 0x7fffffffu;

struct ClusterInfo {
	uvec4 subdivisions;
	uvec2 cluster_size;
	float z_near;
	float k_near;
	float depth_factor;
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
	vec4 lum;
	float radius;
};

struct DirLight {
	vec4 dir;
	vec4 irradiance;
};
