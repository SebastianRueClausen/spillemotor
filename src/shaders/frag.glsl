#version 450
#pragma shader_stage(fragment)

#include "cluster_general.glsl"

#define CLUSTERED

#ifdef DEBUG
#include "cluster_debug.glsl"
#endif

const float PI = 3.14159265358979323846264;

readonly layout (std140, set = 0, binding = 0) uniform View {
	vec4 eye;
	mat4 view;
	mat4 proj_view;
};

readonly layout (std140, set = 0, binding = 1) uniform Proj {
	mat4 proj;
	mat4 inverse_proj;
	vec2 screen_dimensions;
};

layout (set = 0, binding = 2) uniform sampler2D base_color_sampler;
layout (set = 0, binding = 3) uniform sampler2D normal_sampler;
layout (set = 0, binding = 4) uniform sampler2D metallic_roughness_sampler;

readonly layout (std140, set = 1, binding = 0) uniform Cluster {
	ClusterInfo cluster_info;
};

readonly layout (std430, set = 1, binding = 1) buffer Lights {
	uint point_light_count;

	DirLight dir_light;
	PointLight point_lights[];
};

readonly layout (std430, set = 1, binding = 2) buffer LightMasks {
	LightMask light_masks[];
};

layout (location = 0) in vec2 in_texcoord;

layout (location = 1) in vec3 in_world_normal;
layout (location = 2) in vec4 in_world_tangent;
layout (location = 3) in vec3 in_world_bitangent;
layout (location = 4) in vec4 in_world_position;

layout (location = 5) in float in_view_z;

layout (location = 0) out vec4 out_color;

float distribution(float normal_dot_half, float roughness) {
	float a = roughness * roughness;
	float f = (normal_dot_half * a - normal_dot_half) * normal_dot_half + 1.0;
	return a / (PI * f * f);
}

float geometric_occlusion(float normal_dot_view, float normal_dot_light, float roughness) {
	float a = roughness;
	float light = normal_dot_light / (normal_dot_view * (1.0 - a) + a);
	float view = normal_dot_view / (normal_dot_light * (1.0 - a) + a);
	return 0.5 / (view + light);
}

// Faster than `pow(x, 5.0)`.
float pow5(float val) {
	return val * val * val * val * val;
}

vec3 fresnel(float light_dot_half, vec3 f0) {
	float f = pow5(1.0 - light_dot_half);
	return f + f0 * (1.0 - f);
}

// The diffuse term of the BRDF.
const float DIFFUSE_TERM = 1.0 / PI;

/// Bidirectional reflectance distribution function.
vec3 brdf(
	vec3 albedo,
	vec3 normal,
	vec3 view_dir,
	vec3 light_dir,
	vec3 irradiance,
	float roughness,
	float metallic,
	vec3 f0
) {
	vec3 halfway = normalize(light_dir + view_dir);

	float normal_dot_view = clamp(abs(dot(normal, view_dir)), 0.001, 1.0);
	float normal_dot_light = clamp(dot(normal, light_dir), 0.001, 1.0);
	float normal_dot_half = clamp(dot(normal, halfway), 0.0, 1.0);
	float light_dot_half = clamp(dot(light_dir, halfway), 0.0, 1.0);

	float d = distribution(normal_dot_half, roughness);
	vec3 f = fresnel(light_dot_half, f0);
	float v = geometric_occlusion(normal_dot_view, normal_dot_light, roughness);

	vec3 diffuse = (1 - f) * albedo * DIFFUSE_TERM;
	vec3 specular = (d * f * v) / max(4.0 * normal_dot_view * normal_dot_light, 0.001);

	return (diffuse + specular) * irradiance * normal_dot_light;
}

// Approximation of ACES (academy color encoding system) tone mapping used by unreal engine 4.
//
// Source: http://graphics-programming.org/resources/tonemapping/index.html
vec3 aces_approx(vec3 color) {
    color *= 0.6f;

    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;

    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0f, 1.0f);
}

// Specular antialiasing.
//
// Source: http://www.jp.square-enix.com/tech/library/pdf/ImprovedGeometricSpecularAA.pdf
float specular_aa(vec3 normal, float roughness) {
    const float SIGMA2 = 0.25;
    const float KAPPA  = 0.18;

	vec3 dndu = dFdx(normal);
    vec3 dndv = dFdy(normal);

    float variance = SIGMA2 * (dot(dndu, dndu) + dot(dndv, dndv));
    float kernel_roughness = min(2.0 * variance, KAPPA);

    return clamp(roughness + kernel_roughness, 0.0, 1.0);
}

uvec3 cluster_coords(vec2 coords, float view_z) {
	uvec2 ij = uvec2(coords / cluster_info.cluster_size.xy);
	uint k = uint(log(-view_z) * cluster_info.depth_factors.x - cluster_info.depth_factors.y);
	return uvec3(ij, k);
}

void main() {
	vec4 color = texture(base_color_sampler, in_texcoord);
	vec3 normal = texture(normal_sampler, in_texcoord).rgb * 2.0 - 1.0;
	vec2 metallic_roughness = texture(metallic_roughness_sampler, in_texcoord).rg;

	normal = normalize(
		normal.x * in_world_tangent.xyz
			+ normal.y * in_world_bitangent
			+ normal.z * normalize(in_world_normal)
	);

	vec3 albedo = color.rgb;
	float metallic = metallic_roughness.r;
	float roughness = specular_aa(normal, metallic_roughness.g);

	// Reflectance at normal incidence.	
	vec3 f0 = mix(vec3(0.04), albedo, metallic);
	vec3 view_dir = normalize(vec3(eye) - vec3(in_world_position));

	vec3 radiance = brdf(
		albedo,
		normal,
		view_dir,
		vec3(dir_light.dir),
		vec3(dir_light.irradiance),
		roughness,
		metallic,
		f0
	);

#ifdef CLUSTERED
	uvec3 cluster_coords = cluster_coords(gl_FragCoord.xy, in_view_z);
	uint cluster_index = cluster_index(cluster_info.subdivisions.xyz, cluster_coords);
	uint light_count = 0;

	LightMask light_mask = light_masks[cluster_index];
	
	for (uint i = 0; i < LIGHT_MASK_WORD_COUNT; ++i) {
		uint word = light_mask.mask[i];

		while (word != 0) {
			uint bit_index = findLSB(word);
			uint light_index = i * 32 + bit_index;

			word &= ~uint(1 << bit_index);
			light_count += 1;

			PointLight light = point_lights[light_index];

			vec3 light_dir = normalize(vec3(light.pos) - vec3(in_world_position));
			float dist = length(vec3(light.pos) - vec3(in_world_position));
			vec3 irradiance = vec3(light.lum) / (4.0 * PI * dist * dist);

			radiance += brdf(albedo, normal, view_dir, light_dir, irradiance, roughness, metallic, f0);
		}
	}
#else
	for (uint light_index = 0; light_index < point_light_count; light_index++) {
		PointLight light = point_lights[light_index];

		vec3 light_dir = normalize(light.pos - vec3(in_ws_pos));
		float dist = length(light.pos - vec3(in_ws_pos));
		vec3 irradiance = light.lum / (4.0 * PI * dist * dist);

		radiance += brdf(albedo, normal, view_dir, light_dir, irradiance, roughness, metallic, f0);
	}
#endif

	vec3 ambient = vec3(0.040) * albedo;
	radiance += ambient;
	out_color = vec4(aces_approx(radiance), 1.0);

#ifdef CLUSTERED
#ifdef DEBUG
	out_color = debug_cluster_overlay(out_color, cluster_coords, light_count);
#endif
#endif
}
