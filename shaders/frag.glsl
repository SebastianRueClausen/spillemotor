#version 450
#pragma shader_stage(fragment)

const float PI = 3.14159265358979323846264;

struct DirLight {
	vec3 dir;
	vec3 irradiance;
};

struct PointLight {
	vec3 pos;
	vec3 lum;
	float radius;
};

struct ClusterLights {
	uint count;
	uint off;
};

layout (std140, set = 0, binding = 1) uniform Ubo { 
	DirLight dir_light;
	vec3 camera_pos;
} ubo;

layout (set = 0, binding = 2) uniform sampler2D tex_sampler;

readonly layout (std140, set = 1, binding = 0) buffer LightGeneral {
	mat4 perspective_inverse;
	uvec2 screen_extent;
	uvec4 cluster_grid;
	float znear;
	float zfar;
} light_general;

readonly layout (std140, set = 1, binding = 1) buffer LightBuffer {
	uint count;
	uint capacity;
	PointLight lights[];
} light_buffer;

/*
readonly layout (set = 1, binding = 2) buffer ActiveLights {
	uint active_lights[8 * 4 * 8 * 128];
} active_lights;

readonly layout (set = 1, binding = 3) buffer ClusterLights {
	ClusterLights active_lights[128];
} cluster_lights;
*/

layout (push_constant) uniform PushConstant {
	layout(offset = 64) float roughness;
	layout(offset = 68) float metallic;
} pc;

layout (location = 0) in vec2 in_texcoord;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec3 in_ws_pos;

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
//
// `abledo`: The base color of the surface.
// `view_dir`: The direction of the camera.
// `light_dir`: The direction of the light.
// `f0`: The reflectance of the surface at an normal incidence.
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

void main() {
	vec4 color = texture(tex_sampler, in_texcoord);
	vec3 view_dir = normalize(ubo.camera_pos - in_ws_pos);

	vec3 albedo = color.rgb;

	float metallic = pc.metallic;
	float roughness = pc.roughness * pc.roughness;

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
		ubo.dir_light.dir,
		ubo.dir_light.irradiance,
		roughness,
		metallic,
		f0
	);

	for (uint i = 0; i < light_buffer.count; i += 1) {
		PointLight light = light_buffer.lights[i];

		// Direction to the light.
		vec3 light_dir = normalize(light.pos - in_ws_pos);

		// Distance to the light.
		float dist = length(light.pos - in_ws_pos);

		// The irradiance of the light given it's lumen and distance. 
		vec3 irradiance = light.lum / (4.0 * PI * dist * dist);

		radiance += brdf(albedo, normal, view_dir, light_dir, irradiance, roughness, metallic, f0);
	}

	vec3 ambient = vec3(0.040) * albedo;

	radiance += ambient;
	out_color = vec4(aces_approx(radiance), 1.0);
}
