#version 450
#pragma shader_stage(vertex)

layout(set = 0, binding = 0) uniform UniformBufferObject {
	mat4 perspective;
	mat4 view;
} ubo;

layout (location=0) in vec3 position;
layout (location=0) out vec4 frag_color;

void main() {
	gl_Position = ubo.perspective * ubo.view * vec4(position, 1.0);
	frag_color = vec4(1.0, 1.0, 1.0, 1.0);
}
