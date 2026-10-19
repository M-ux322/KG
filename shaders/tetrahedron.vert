#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 0) out vec3 fragColor;
layout(set = 0, binding = 0) uniform ObjectData {
    mat4 model;
    vec4 tint;
} objectData;
layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 colorMultiplier;
} pc;
void main() {
    gl_Position = pc.mvp * objectData.model * vec4(inPosition, 1.0);
    fragColor = inColor;
}
