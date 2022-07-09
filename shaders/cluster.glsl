#version 450
#pragma shader_stage(compute)

struct ClusterAabb {
	vec3 min;
	vec3 max;
};

layout (std140, set = 0, binding = 0) buffer General {
		mat4 perspective_inverse;
		uvec2 screen_extent;
		uvec4 cluster_grid;
} general;

layout (std140, set = 0, binding = 0) buffer AabbBuffer {
	ClusterAabb aabs[];
} aabb_buffer;

void main() {

}
