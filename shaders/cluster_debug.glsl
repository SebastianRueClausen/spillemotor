
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

    float max_light_complexity_per_cluster = 1.0;
    background.r = (1.0 - cluster_overlay_alpha) * background.r
        + cluster_overlay_alpha * smoothstep(0.0, max_light_complexity_per_cluster, float(lights_in_cluster));
    background.g = (1.0 - cluster_overlay_alpha) * background.g
        + cluster_overlay_alpha * (1.0 - smoothstep(0.0, max_light_complexity_per_cluster, float(lights_in_cluster)));

	return background;
}

