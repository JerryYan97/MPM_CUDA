//
// Created by jiaruiyan on 5/24/21.
//

#include "graphicsPipeline.h"
#include <streambuf>
#include <sstream>
#include <filesystem>
#include <glad/glad.h>
#include <fstream>

graphicsPipeline::graphicsPipeline(std::string &vPath, std::string &fPath) {
    std::ifstream vert_shader_handle(vPath);
    std::stringstream vert_ss;
    vert_ss << vert_shader_handle.rdbuf();
    std::string vert_shader_source = vert_ss.str();
    const char* vert_source_ptr = vert_shader_source.c_str();
    vert_shader_handle.close();

    std::ifstream frag_shader_handle(fPath);
    std::stringstream frag_ss;
    frag_ss << frag_shader_handle.rdbuf();
    std::string frag_shader_source = frag_ss.str();
    const char* frag_source_ptr = frag_shader_source.c_str();
    frag_shader_handle.close();

    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vert_source_ptr, NULL);
    glCompileShader(vertexShader);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &frag_source_ptr, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();

    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    ID = shaderProgram;
}

void graphicsPipeline::use() {
    glUseProgram(ID);
}

void graphicsPipeline::destroy() {
    glDeleteProgram(ID);
}

void graphicsPipeline::setVec4(const std::string &name, std::array<float, 4> &val) {
    int location = glGetUniformLocation(ID, name.c_str());
    this->use();
    glUniform4f(location, val[0], val[1], val[2], val[3]);
}

void graphicsPipeline::setMat4(const std::string &name, glm::mat4 &val) {
    int location = glGetUniformLocation(ID, name.c_str());
    this->use();
    glUniformMatrix4fv(location, 1, GL_FALSE, &val[0][0]);
}

void graphicsPipeline::setVec3(const std::string &name, std::array<float, 3> &val) {
    int location = glGetUniformLocation(ID, name.c_str());
    this->use();
    glUniform3f(location, val[0], val[1], val[2]);
}

void graphicsPipeline::setMat3(const std::string &name, glm::mat3 &val) {
    int location = glGetUniformLocation(ID, name.c_str());
    this->use();
    glUniformMatrix3fv(location, 1, GL_FALSE, &val[0][0]);
}
