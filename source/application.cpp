#include "application.hpp"

#include <imgui.h>
#include <array>
#include <vector>
#include <cstdint>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <cstddef>
#include <string>
#include <cmath>
#include <array>

namespace application {

float rotationAngle = 0.0f;

float rotationSpeed = 0.7f;
bool autoRotate = true;

float objectScale = 0.38f;

float positionX = 0.0f;
float positionY = 0.0f;
float positionZ = 0.0f;

float manualRotationX = 0.0f;
float manualRotationY = 0.0f;
float manualRotationZ = 0.0f;

float stretchX = 1.0f;
float stretchY = 1.0f;
float stretchZ = 1.0f;

bool trajectoryEnabled = true;

float trajectoryTime = 0.0f;
float trajectorySpeed = 0.7f;
float trajectoryRadius = 0.35f;
float trajectoryHeight = 0.3f;

float figureColor[3] = {1.0f, 1.0f, 1.0f};

using Matrix4 = std::array<float, 16>;

bool perspectiveProjection = true;

struct Vertex {
    float position[3];
    float color[3];
};

std::vector<Vertex> vertices;
std::vector<uint32_t> indices;

VkBuffer vertexBuffer = VK_NULL_HANDLE;
VmaAllocation vertexAllocation = VK_NULL_HANDLE;

VkBuffer indexBuffer = VK_NULL_HANDLE;
VmaAllocation indexAllocation = VK_NULL_HANDLE;

VkPipeline graphicsPipeline = VK_NULL_HANDLE;
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

// One uniform buffer and descriptor set for each tetrahedron.
struct alignas(16) ObjectUniform {
    Matrix4 model;
    float tint[4];
};

VkDescriptorSetLayout objectDescriptorLayout = VK_NULL_HANDLE;
VkDescriptorPool objectDescriptorPool = VK_NULL_HANDLE;
VkDescriptorSet objectDescriptorSets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
VkBuffer objectUniformBuffers[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
VmaAllocation objectUniformAllocations[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

void createTruncatedTetrahedron() {
    vertices.clear();
    indices.clear();

    const std::array<std::array<float, 3>, 4> original = {{
        {{ 1.0f,  1.0f,  1.0f}},
        {{ 1.0f, -1.0f, -1.0f}},
        {{-1.0f,  1.0f, -1.0f}},
        {{-1.0f, -1.0f,  1.0f}}
    }};

    // 12 точек на рёбрах исходного тетраэдра.
    std::array<std::array<float, 3>, 12> points{};
    int pointIndex[4][4] = {};

    int nextIndex = 0;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j) {
                continue;
            }

            pointIndex[i][j] = nextIndex;

            for (int k = 0; k < 3; ++k) {
                points[nextIndex][k] =
                    (2.0f * original[i][k] +
                     original[j][k]) / 3.0f;
            }

            ++nextIndex;
        }
    }

    auto addFace = [&](const std::vector<int>& face,
                       int colorIndex) {
        const uint32_t start =
            static_cast<uint32_t>(vertices.size());


		for (int point : face) {
			Vertex vertex{};

			for (int k = 0; k < 3; ++k) {
				vertex.position[k] = points[point][k];

				// Цвет зависит от координаты вершины.
				vertex.color[k] = (points[point][k] + 1.0f) * 0.5f;
			}

			vertices.push_back(vertex);
		}

        for (uint32_t k = 1;
             k + 1 < static_cast<uint32_t>(face.size());
             ++k) {
            indices.push_back(start);
            indices.push_back(start + k);
            indices.push_back(start + k + 1);
        }
    };

    for (int i = 0; i < 4; ++i) {
        std::vector<int> face;

        for (int j = 0; j < 4; ++j) {
            if (i != j) {
                face.push_back(pointIndex[i][j]);
            }
        }

        addFace(face, i);
    }

    const int originalFaces[4][3] = {
        {0, 1, 2},
        {0, 3, 1},
        {0, 2, 3},
        {1, 3, 2}
    };

    for (int f = 0; f < 4; ++f) {
        const int a = originalFaces[f][0];
        const int b = originalFaces[f][1];
        const int c = originalFaces[f][2];

        addFace({
            pointIndex[a][b],
            pointIndex[b][a],
            pointIndex[b][c],
            pointIndex[c][b],
            pointIndex[c][a],
            pointIndex[a][c]
        }, f + 4);
    }
}

bool createBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    const void* data,
    VkBuffer& buffer,
    VmaAllocation& allocation
) {
    auto& context = graphics::internal::context;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    if (vmaCreateBuffer(
            context.allocator,
            &bufferInfo,
            &allocationInfo,
            &buffer,
            &allocation,
            nullptr
        ) != VK_SUCCESS) {
        return false;
    }

    void* mappedData = nullptr;

    if (vmaMapMemory(
            context.allocator,
            allocation,
            &mappedData
        ) != VK_SUCCESS) {
        vmaDestroyBuffer(context.allocator, buffer, allocation);
        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
        return false;
    }

    std::memcpy(mappedData, data, static_cast<size_t>(size));
    vmaFlushAllocation(context.allocator, allocation, 0, size);
    vmaUnmapMemory(context.allocator, allocation);

    return true;
}

bool createGeometryBuffers() {
    const VkDeviceSize vertexSize =
        sizeof(Vertex) * vertices.size();

    const VkDeviceSize indexSize =
        sizeof(uint32_t) * indices.size();

    if (!createBuffer(
            vertexSize,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vertices.data(),
            vertexBuffer,
            vertexAllocation
        )) {
        return false;
    }

    if (!createBuffer(
            indexSize,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            indices.data(),
            indexBuffer,
            indexAllocation
        )) {
        auto& context = graphics::internal::context;

        vmaDestroyBuffer(
            context.allocator,
            vertexBuffer,
            vertexAllocation
        );

        vertexBuffer = VK_NULL_HANDLE;
        vertexAllocation = VK_NULL_HANDLE;

        return false;
    }

    return true;
}


std::vector<char> readShaderFile(const char* filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error(
            std::string("Cannot open shader: ") + filename
        );
    }

    const size_t fileSize =
        static_cast<size_t>(file.tellg());

    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    return buffer;
}

VkShaderModule createShaderModule(
    const std::vector<char>& code
) {
    auto& context = graphics::internal::context;

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

    createInfo.codeSize = code.size();
    createInfo.pCode =
        reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;

    if (vkCreateShaderModule(
            context.device,
            &createInfo,
            nullptr,
            &shaderModule
        ) != VK_SUCCESS) {
        throw std::runtime_error(
            "Failed to create shader module"
        );
    }

    return shaderModule;
}

bool createObjectDescriptors() {
    auto& context = graphics::internal::context;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr,
                                    &objectDescriptorLayout) != VK_SUCCESS) return false;

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 2;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr,
                               &objectDescriptorPool) != VK_SUCCESS) return false;

    VkDescriptorSetLayout layouts[2] = {objectDescriptorLayout, objectDescriptorLayout};
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = objectDescriptorPool;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(context.device, &allocInfo,
                                 objectDescriptorSets) != VK_SUCCESS) return false;

    for (int i = 0; i < 2; ++i) {
        ObjectUniform data{};
        data.model = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            i == 0 ? -2.3f : 2.3f, 0, 0, 1
        };
        if (i == 0) {
            data.tint[0] = 1.0f; data.tint[1] = 1.0f;
            data.tint[2] = 1.0f; data.tint[3] = 1.0f;
        } else {
            data.tint[0] = 1.0f; data.tint[1] = 0.65f;
            data.tint[2] = 0.75f; data.tint[3] = 1.0f;
        }
        if (!createBuffer(sizeof(data), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          &data, objectUniformBuffers[i],
                          objectUniformAllocations[i])) return false;

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = objectUniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(data);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = objectDescriptorSets[i];
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &bufferInfo;
        vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    }
    return true;
}

bool createGraphicsPipeline() {
    auto& context = graphics::internal::context;

    const auto vertCode =
        readShaderFile("shaders/tetrahedron.vert.spv");

    const auto fragCode =
        readShaderFile("shaders/tetrahedron.frag.spv");

    VkShaderModule vertModule =
        createShaderModule(vertCode);

    VkShaderModule fragModule =
        createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        vertStage, fragStage
    };

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributes[2]{};

    attributes[0].binding = 0;
    attributes[0].location = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = offsetof(Vertex, position);

    attributes[1].binding = 0;
    attributes[1].location = 1;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = offsetof(Vertex, color);

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 2;
    vertexInput.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology =
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorAttachment{};
    colorAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

	VkPushConstantRange pushConstant{};
	pushConstant.stageFlags =
		VK_SHADER_STAGE_VERTEX_BIT |
		VK_SHADER_STAGE_FRAGMENT_BIT;

	pushConstant.offset = 0;
	pushConstant.size = sizeof(Matrix4) + sizeof(float) * 4;

	VkPipelineLayoutCreateInfo layoutInfo{};
	layoutInfo.sType =
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layoutInfo.pushConstantRangeCount = 1;
	layoutInfo.pPushConstantRanges = &pushConstant;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &objectDescriptorLayout;

    if (vkCreatePipelineLayout(
            context.device,
            &layoutInfo,
            nullptr,
            &pipelineLayout
        ) != VK_SUCCESS) {
        vkDestroyShaderModule(context.device, fragModule, nullptr);
        vkDestroyShaderModule(context.device, vertModule, nullptr);
        return false;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType =
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = context.render_pass;
    pipelineInfo.subpass = 0;

    const VkResult result = vkCreateGraphicsPipelines(
        context.device,
        VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &graphicsPipeline
    );

    vkDestroyShaderModule(context.device, fragModule, nullptr);
    vkDestroyShaderModule(context.device, vertModule, nullptr);

    return result == VK_SUCCESS;
}

bool initialize() {
    createTruncatedTetrahedron();

    return createGeometryBuffers()
        && createObjectDescriptors()
        && createGraphicsPipeline();
}

void shutdown() {
    auto& context = graphics::internal::context;

    vkQueueWaitIdle(context.graphics_queue);

	if (graphicsPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(
        context.device,
        graphicsPipeline,
        nullptr
    );
}

if (pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(
        context.device,
        pipelineLayout,
        nullptr
    );
}

    for (int i = 0; i < 2; ++i) {
        if (objectUniformBuffers[i] != VK_NULL_HANDLE) {
            vmaDestroyBuffer(context.allocator, objectUniformBuffers[i],
                             objectUniformAllocations[i]);
        }
    }
    if (objectDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(context.device, objectDescriptorPool, nullptr);
    }
    if (objectDescriptorLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(context.device, objectDescriptorLayout, nullptr);
    }

    if (indexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            indexBuffer,
            indexAllocation
        );
    }

    if (vertexBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            vertexBuffer,
            vertexAllocation
        );
    }
}

void update(double time) {
    static double previousTime = time;

    double deltaTime = time - previousTime;
    previousTime = time;

    if (deltaTime < 0.0 || deltaTime > 0.1) {
        deltaTime = 0.0;
    }

    if (autoRotate) {
        rotationAngle +=
            rotationSpeed * static_cast<float>(deltaTime);
    }

	if (trajectoryEnabled) {
    trajectoryTime +=
        trajectorySpeed * static_cast<float>(deltaTime);
	}

    ImGui::Begin("Lab 1 - Truncated Tetrahedron");

    ImGui::Text("Truncated regular tetrahedron");
    ImGui::Separator();

    ImGui::Text("Rotation");
    ImGui::Checkbox("Auto rotation", &autoRotate);
    ImGui::SliderFloat(
        "Rotation speed",
        &rotationSpeed,
        0.0f,
        3.0f
    );

    ImGui::Separator();

    ImGui::Text("Scale");
    ImGui::SliderFloat(
        "Object scale",
        &objectScale,
        0.1f,
        0.8f
    );

    ImGui::Separator();

    ImGui::Text("Position");
    ImGui::SliderFloat("X", &positionX, -0.5f, 0.5f);
    ImGui::SliderFloat("Y", &positionY, -0.5f, 0.5f);
    ImGui::SliderFloat("Z", &positionZ, -0.5f, 0.5f);

	ImGui::Separator();
	ImGui::Text("Manual rotation (degrees)");

	ImGui::SliderFloat("Rotation X", &manualRotationX, -180.0f, 180.0f);
	ImGui::SliderFloat("Rotation Y", &manualRotationY, -180.0f, 180.0f);
	ImGui::SliderFloat("Rotation Z", &manualRotationZ, -180.0f, 180.0f);

	ImGui::Separator();
	ImGui::Text("Stretch");

	ImGui::SliderFloat("Stretch X", &stretchX, 0.2f, 3.0f);
	ImGui::SliderFloat("Stretch Y", &stretchY, 0.2f, 3.0f);
	ImGui::SliderFloat("Stretch Z", &stretchZ, 0.2f, 3.0f);

	ImGui::Separator();

	ImGui::Separator();
	ImGui::Text("Trajectory animation");

	ImGui::Checkbox("Play trajectory", &trajectoryEnabled);

	ImGui::SliderFloat(
		"Trajectory speed",
		&trajectorySpeed,
		0.0f,
		3.0f
	);

	ImGui::SliderFloat(
		"Trajectory radius",
		&trajectoryRadius,
		0.0f,
		1.5f
	);

	ImGui::SliderFloat(
		"Trajectory height",
		&trajectoryHeight,
		0.0f,
		1.0f
	);

	if (ImGui::Button("Restart trajectory")) {
		trajectoryTime = 0.0f;
	}

	ImGui::Separator();
	ImGui::Text("Figure color");

	ImGui::ColorEdit3("Color", figureColor);

	ImGui::Text("Projection");

	if (ImGui::RadioButton("Perspective", perspectiveProjection)) {
    perspectiveProjection = true;
	}

	ImGui::SameLine();

	if (ImGui::RadioButton("Orthographic", !perspectiveProjection)) {
		perspectiveProjection = false;
	}

    if (ImGui::Button("Reset")) {
        rotationAngle = 0.0f;
        rotationSpeed = 0.7f;
        autoRotate = true;
		perspectiveProjection = true;

        objectScale = 0.38f;

        positionX = 0.0f;
        positionY = 0.0f;
        positionZ = 0.0f;

		manualRotationX = 0.0f;
		manualRotationY = 0.0f;
		manualRotationZ = 0.0f;

		stretchX = 1.0f;
		stretchY = 1.0f;
		stretchZ = 1.0f;

		trajectoryEnabled = true;
		trajectoryTime = 0.0f;
		trajectorySpeed = 0.7f;
		trajectoryRadius = 0.35f;
		trajectoryHeight = 0.3f;

		figureColor[0] = 1.0f;
		figureColor[1] = 1.0f;
		figureColor[2] = 1.0f;
    }

    ImGui::End();
}

Matrix4 multiplyMatrices(
    const Matrix4& a,
    const Matrix4& b
) {
    Matrix4 result{};

    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            for (int k = 0; k < 4; ++k) {
                result[col * 4 + row] +=
                    a[k * 4 + row] *
                    b[col * 4 + k];
            }
        }
    }

    return result;
}

Matrix4 createProjection(
    bool perspective,
    float aspect
) {
    Matrix4 result{};

    const float nearPlane = 0.1f;
    const float farPlane = 100.0f;

    if (perspective) {
        const float fov = 45.0f * 3.14159265f / 180.0f;
        const float f = 1.0f / std::tan(fov / 2.0f);

        result[0] = f / aspect;
        result[5] = -f;
        result[10] =
            farPlane / (nearPlane - farPlane);
        result[11] = -1.0f;
        result[14] =
            (farPlane * nearPlane) /
            (nearPlane - farPlane);
    } else {
        const float height = 2.5f;
        const float width = height * aspect;

        result[0] = 2.0f / width;
        result[5] = -2.0f / height;
        result[10] = 1.0f / (nearPlane - farPlane);
        result[14] =
            nearPlane / (nearPlane - farPlane);
        result[15] = 1.0f;
    }

    return result;
}


void render(const graphics::internal::FrameData& fd) {
    auto& context = graphics::internal::context;

    if (vkResetCommandBuffer(fd.command_buffer, 0) != VK_SUCCESS) {
        return;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(fd.command_buffer, &beginInfo) != VK_SUCCESS) {
        return;
    }

    VkClearValue clearValues[2]{};
    clearValues[0].color = {{0.12f, 0.16f, 0.23f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = context.render_pass;
    renderPassInfo.framebuffer = fd.framebuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = context.swapchain_extent;
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(
        fd.command_buffer,
        &renderPassInfo,
        VK_SUBPASS_CONTENTS_INLINE
    );

    
	vkCmdBindPipeline(
		fd.command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		graphicsPipeline
	);

	VkViewport viewport{};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width =
		static_cast<float>(context.swapchain_extent.width);
	viewport.height =
		static_cast<float>(context.swapchain_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.offset = {0, 0};
	scissor.extent = context.swapchain_extent;

	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	VkBuffer buffers[] = {vertexBuffer};
	VkDeviceSize offsets[] = {0};

	vkCmdBindVertexBuffers(
		fd.command_buffer,
		0,
		1,
		buffers,
		offsets
	);

	vkCmdBindIndexBuffer(
		fd.command_buffer,
		indexBuffer,
		0,
		VK_INDEX_TYPE_UINT32
	);

	const float toRadians = 3.14159265f / 180.0f;

	const float ax = manualRotationX * toRadians
				+ rotationAngle * 0.6f;
	const float ay = manualRotationY * toRadians
				+ rotationAngle;
	const float az = manualRotationZ * toRadians;

	const float cx = std::cos(ax);
	const float sx = std::sin(ax);

	const float cy = std::cos(ay);
	const float sy = std::sin(ay);

	const float cz = std::cos(az);
	const float sz = std::sin(az);

	const Matrix4 scaling = {
		objectScale * stretchX, 0, 0, 0,
		0, objectScale * stretchY, 0, 0,
		0, 0, objectScale * stretchZ, 0,
		0, 0, 0, 1
	};

	const Matrix4 rotationX = {
		1, 0, 0, 0,
		0, cx, sx, 0,
		0, -sx, cx, 0,
		0, 0, 0, 1
	};

	const Matrix4 rotationY = {
		cy, 0, -sy, 0,
		0, 1, 0, 0,
		sy, 0, cy, 0,
		0, 0, 0, 1
	};

	const Matrix4 rotationZ = {
		cz, sz, 0, 0,
		-sz, cz, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	};

	const float t = trajectoryTime;

	const float trajectoryX =
		trajectoryRadius * std::sin(t);

	const float trajectoryY =
		trajectoryRadius * 0.5f * std::sin(2.0f * t);

	const float trajectoryZ =
		trajectoryHeight * std::cos(t);

	const Matrix4 translation = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,

		positionX + trajectoryX,
		positionY + trajectoryY,
		positionZ + trajectoryZ,
		1
	};

	const Matrix4 model = multiplyMatrices(
		translation,
		multiplyMatrices(
			rotationZ,
			multiplyMatrices(
				rotationY,
				multiplyMatrices(rotationX, scaling)
			)
		)
	);

	const Matrix4 view = {
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, -4.0f, 1.0f
	};

	const float aspect =
		static_cast<float>(context.swapchain_extent.width) /
		static_cast<float>(context.swapchain_extent.height);

	const Matrix4 projection =
		createProjection(perspectiveProjection, aspect);

	// GLSL: gl_Position = projection * view * model * position
	const Matrix4 mvp = multiplyMatrices(
		projection,
		multiplyMatrices(view, model)
	);

	struct PushConstants {
		Matrix4 mvp;
		float colorMultiplier[4];
	};

	const PushConstants pushConstants = {
		mvp,
		{
			figureColor[0],
			figureColor[1],
			figureColor[2],
			1.0f
		}
	};

    for (int i = 0; i < 2; ++i) {
        vkCmdBindDescriptorSets(
            fd.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout,
            0, 1, &objectDescriptorSets[i],
            0, nullptr
        );
        vkCmdPushConstants(
            fd.command_buffer,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(PushConstants), &pushConstants
        );
        vkCmdDrawIndexed(
            fd.command_buffer,
            static_cast<uint32_t>(indices.size()),
            1, 0, 0, 0
        );
    }

    vkCmdEndRenderPass(fd.command_buffer);

    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application