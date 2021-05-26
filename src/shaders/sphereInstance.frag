#version 330 core
out vec4 FragColor;
in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightColor;
uniform vec3 objColor;
uniform vec3 lightPos;
uniform vec3 camPos;

void main()
{
    float ambientStrength = 0.1;
    float specularStrength = 0.5;

    vec3 ambient = ambientStrength * lightColor;

    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(Normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 viewDir = normalize(camPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, Normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objColor;
    FragColor = vec4(result, 1.0);
}