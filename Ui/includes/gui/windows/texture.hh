/*! \file Texture.hh
 *  \brief Class for creating and managing an OpenGL texture with CUDA interoperability.
 */

#pragma once

#ifdef WIN32
#include <windows.h>
#endif
#include <cuda_gl_interop.h>
#include "BasicOpenGLWindow.hh"

/*! \class Texture
 *  \brief Encapsulates the creation and management of an OpenGL texture with CUDA interoperability.
 *
 *  This class handles the allocation, binding, updating, and CUDA mapping/unmapping of an OpenGL texture.
 */
class Texture
{
  public:
    /*! \brief Constructor.
     *
     *  Creates and configures an OpenGL texture according to the specified parameters.
     *
     *  \param target The texture target (e.g., GL_TEXTURE_2D).
     *  \param width The width of the texture.
     *  \param height The height of the texture.
     *  \param internalFormat The internal format of the texture (e.g., GL_RGBA).
     *  \param format The format of the provided data (e.g., GL_RGBA).
     *  \param type The data type (e.g., GL_UNSIGNED_BYTE).
     */
    Texture(GLenum target, int width, int height, GLenum internalFormat, GLenum format, GLenum type);

    /*! \brief Destructor.
     *
     *  Releases CUDA and OpenGL resources.
     */
    ~Texture();

    /*! \brief Binds the texture for OpenGL usage.
     */
    void bind() const;

    /*! \brief Unbinds the texture.
     */
    void unbind() const;

    /*! \brief Updates the texture with new data.
     *
     *  \param data Pointer to the data to be copied into the texture.
     */
    void updateData(void* data);

    /*! \brief Maps the texture for direct CUDA access.
     *
     *  \param stream The CUDA stream used for mapping.
     *  \return The CUDA array associated with the texture.
     */
    cudaArray_t mapForCuda(cudaStream_t stream);

    /*! \brief Unmaps the texture after CUDA access.
     *
     *  \param stream The CUDA stream used for unmapping.
     */
    void unmapFromCuda(cudaStream_t stream);

    /*! \brief Retrieves the OpenGL texture ID.
     *
     *  \return The texture ID.
     */
    GLuint getTextureID() const;

    /*! \brief Retrieves the texture width.
     *
     *  \return The width of the texture.
     */
    int getWidth() const;

    /*! \brief Retrieves the texture height.
     *
     *  \return The height of the texture.
     */
    int getHeight() const;

  private:
    GLuint textureID;      // OpenGL texture identifier.
    GLenum target;         // OpenGL texture target (e.g., GL_TEXTURE_2D).
    int width;             // Texture width.
    int height;            // Texture height.
    GLenum internalFormat; // Internal format of the texture.
    GLenum format;         // Format of the data provided.
    GLenum type;           // Data type of the provided data.

    cudaGraphicsResource* cudaResource; // CUDA resource associated with the texture.

    /*! \brief Registers the texture with CUDA for direct access.
     *
     *  This method registers the texture using cudaGraphicsGLRegisterImage with the
     *  cudaGraphicsRegisterFlagsWriteDiscard flag.
     */
    void registerCudaResource();
};