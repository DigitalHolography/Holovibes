/*! \file Texture.cc
 *  \brief Implementation of the Texture class for managing an OpenGL texture with CUDA interoperability.
 */

#include "texture.hh"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWindow>

Texture::Texture(GLenum target, int width, int height, GLenum internalFormat, GLenum format, GLenum type)
    : target(target)
    , width(width)
    , height(height)
    , internalFormat(internalFormat)
    , format(format)
    , type(type)
    , cudaResource(nullptr)
{
    // Generate a texture ID and bind the texture for configuration.
    glGenTextures(1, &textureID);
    glBindTexture(target, textureID);

    // Allocate texture memory without providing initial data.
    glTexImage2D(target, 0, internalFormat, width, height, 0, format, type, nullptr);

    // Set texture wrapping parameters.
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // Set texture filtering parameters.
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Unbind the texture after configuration.
    glBindTexture(target, 0);

    // Register the texture with CUDA.
    registerCudaResource();
}

Texture::~Texture()
{
    if (cudaResource)
    {
        cudaGraphicsUnregisterResource(cudaResource);
    }
    glDeleteTextures(1, &textureID);
}

void Texture::bind() const { glBindTexture(target, textureID); }

void Texture::unbind() const { glBindTexture(target, 0); }

void Texture::updateData(void* data)
{
    bind();
    // Update the texture content.
    glTexSubImage2D(target, 0, 0, 0, width, height, format, type, data);
    // Regenerate mipmaps if necessary.
    // glGenerateMipmap(target);
    unbind();
}

cudaArray_t Texture::mapForCuda(cudaStream_t stream)
{
    cudaGraphicsMapResources(1, &cudaResource, stream);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);
    return array;
}

void Texture::unmapFromCuda(cudaStream_t stream) { cudaGraphicsUnmapResources(1, &cudaResource, stream); }

GLuint Texture::getTextureID() const { return textureID; }

int Texture::getWidth() const { return width; }

int Texture::getHeight() const { return height; }

void Texture::registerCudaResource()
{
    cudaError_t err =
        cudaGraphicsGLRegisterImage(&cudaResource, textureID, target, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        std::cerr << "Error registering CUDA texture resource: " << cudaGetErrorString(err) << std::endl;
    }
}