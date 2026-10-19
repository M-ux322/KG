
#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ObjectData {
    mat4 model;
    vec4 tint;
} objectData;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 colorMultiplier;
} pc;

void main() {
    outColor = vec4(
        fragColor * pc.colorMultiplier.rgb * objectData.tint.rgb,
        1.0
    );
}
