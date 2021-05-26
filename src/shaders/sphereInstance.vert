#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 instancePos;

out vec3 Normal;
out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform mat3 normalMat;

void main()
{
    gl_Position = proj * view * (model * vec4(aPos, 1.0) + vec4(instancePos, 0.0));
    Normal = normalize(normalMat * aNormal);
    FragPos =vec3(model * vec4(aPos, 1.0));
}