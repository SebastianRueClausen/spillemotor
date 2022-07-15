#version 450
#pragma shader_stage(fragment)

#include "cluster_general.glsl"

#define CLUSTERED

const float PI = 3.14159265358979323846264;

readonly layout (std140, set = 0, binding = 0) uniform View {
	vec4 eye;
	mat4 view;
	mat4 inverse_view;
	mat4 proj_view;
};

readonly layout (std140, set = 0, binding = 1) uniform Proj {
	mat4 proj;
	mat4 inverse_proj;
	vec2 screen_dimensions;
};

layout (set = 0, binding = 2) uniform sampler2D tex_sampler;

readonly layout (std140, set = 1, binding = 0) uniform Cluster {
	ClusterInfo cluster_info;
};

readonly layout (std430, set = 1, binding = 1) buffer Lights {
	uint point_light_count;

	DirLight dir_light;
	PointLight point_lights[];
};

readonly layout (std430, set = 1, binding = 2) buffer LightIndices {
	uint light_indices[];
};

layout (push_constant) uniform PushConstant {
	layout(offset = 64) float roughness;
	layout(offset = 68) float metallic;
};

layout (location = 0) in vec2 in_texcoord;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec4 in_ws_pos;
layout (location = 3) in float in_z_view;

layout (location = 0) out vec4 out_color;

// Specular distribution.
//
// This describes the distribution of microfacets of the surface. The algorithm is based on
// trowbridge-reitz.
float distribution(float normal_dot_half, float roughness) {
	float a = roughness * roughness;
	float f = (normal_dot_half * a - normal_dot_half) * normal_dot_half + 1.0;
	return a / (PI * f * f);
}

// Calculate geometric occlusion.
//
// Geometric occlusion is the shadowing from the microfacets of the surface. Basically rough
// surfaces reflects less light back.
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

// The fresnel function.
//
// This calculates the amount of light that reflects from a surface at an incident.
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

uvec3 cluster_coords(vec2 coords, float view_z) {
	uvec2 ij = uvec2(coords / cluster_info.cluster_size);
	uint k = uint(log(-view_z / cluster_info.z_near) * cluster_info.depth_factor);

	return uvec3(ij, k);
}

uint cluster_index(uvec3 coords) {
	return coords.x + cluster_info.subdivisions.x * (coords.y + cluster_info.subdivisions.y * coords.z);
}

#ifdef DEBUG
vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec4 debug_cluster_overlay(vec4 background, uvec3 cluster_coords, uint lights_in_cluster) {
    float cluster_overlay_alpha = 0.3;

	uint z_slice = cluster_coords.z;
    // A hack to make the colors alternate a bit more
    if ((z_slice & 1u) == 1u) {
        z_slice = z_slice + cluster_info.subdivisions.z / 2u;
    }
    vec3 slice_color = hsv2rgb(vec3(float(z_slice) / float(cluster_info.subdivisions.z + 1u), 1.0, 0.5));

	background = vec4(
        (1.0 - cluster_overlay_alpha) * background.rgb + cluster_overlay_alpha * slice_color,
        background.a
    );

	uint cluster_index = cluster_index(cluster_coords);

    vec3 cluster_color = hsv2rgb(vec3(rand(vec2(cluster_index)), 1.0, 0.5));
    background = vec4(
        (1.0 - cluster_overlay_alpha) * background.rgb + cluster_overlay_alpha * cluster_color,
        background.a
    );
	*/

    float max_light_complexity_per_cluster = 1.0;
    background.r = (1.0 - cluster_overlay_alpha) * background.r
        + cluster_overlay_alpha * smoothstep(0.0, max_light_complexity_per_cluster, float(lights_in_cluster));
    background.g = (1.0 - cluster_overlay_alpha) * background.g
        + cluster_overlay_alpha * (1.0 - smoothstep(0.0, max_light_complexity_per_cluster, float(lights_in_cluster)));

	return background;
}

#endif

void main() {
	vec4 color = texture(tex_sampler, in_texcoord);
	vec3 view_dir = normalize(vec3(eye) - vec3(in_ws_pos));

	vec3 albedo = color.rgb;

	float metallic = metallic;
	float roughness = roughness * roughness;

	// Make sure to re-normalize the `in_normal` since interpolating the normals will ever so slightly
	// change the length of the vector.
	//
	// TODO: Not necessary if we do normal mapping.
	vec3 normal = normalize(in_normal);

	// Reflectance at normal incidence.	
	vec3 f0 = mix(vec3(0.04), albedo, metallic);

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

	uvec3 cluster_coords = cluster_coords(gl_FragCoord.xy, in_z_view);
	uint cluster_index = cluster_index(cluster_coords);

#ifdef CLUSTERED

	uint light_base = cluster_index * MAX_LIGHTS_IN_CLUSTER;
	uint light_count = 0;

	for (light_count = 0; light_count < MAX_LIGHTS_IN_CLUSTER; light_count++) {
		uint light_index = light_indices[light_base + light_count];
		if (light_index == LIGHT_INDEX_SENTINEL) {
			break;
		}

		PointLight light = point_lights[light_index];

		vec3 light_dir = normalize(vec3(light.pos) - vec3(in_ws_pos));
		float dist = length(vec3(light.pos) - vec3(in_ws_pos));

		// The irradiance of the light given it's lumen and distance. 
		vec3 irradiance = vec3(light.lum) / (4.0 * PI * dist * dist);

		radiance += brdf(albedo, normal, view_dir, light_dir, irradiance, roughness, metallic, f0);
	}

#else

	for (uint light_index = 0; light_index < point_light_count; light_index++) {
		PointLight light = point_lights[light_index];

		vec3 light_dir = normalize(light.pos - vec3(in_ws_pos));
		float dist = length(light.pos - vec3(in_ws_pos));

		// The irradiance of the light given it's lumen and distance. 
		vec3 irradiance = light.lum / (4.0 * PI * dist * dist);

		radiance += brdf(albedo, normal, view_dir, light_dir, irradiance, roughness, metallic, f0);
	}

#endif

	vec3 ambient = vec3(0.040) * albedo;

	radiance += ambient;
	out_color = vec4(aces_approx(radiance), 1.0);

#ifdef DEBUG
	out_color = debug_cluster_overlay(out_color, cluster_coords, light_count);
#endif
}
