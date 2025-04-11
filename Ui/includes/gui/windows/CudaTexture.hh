/*! \file
 *
 * \brief Cuda texture class used to render the image streams in the Qt windows.
 */
#pragma once

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWindow>

#include "BasicOpenGLWindow.hh"
#include "overlay_manager.hh"
#include "tools_conversion.cuh"
#include "cuda_memory.cuh"
#include "common.cuh"
#include "display_queue.hh"
#include "frame_desc.hh"

namespace holovibes::gui
{

/*! \class CudaTexture
 *  \brief Encapsulates the initialization and management of an OpenGL texture and its CUDA interoperability.
 *
 *  This class is responsible for creating an OpenGL texture, registering it with CUDA,
 *  and providing a method to update the texture using CUDA functions.
 */
class CudaTexture
{
  public:
    /*! \brief Constructor.
     *  \param width Texture width.
     *  \param height Texture height.
     *  \param depth Pixel depth type (used for swizzling).
     *  \param stream Optional CUDA stream for asynchronous operations.
     */
    CudaTexture(int width, int height, camera::PixelDepth depth, cudaStream_t stream = 0);

    /*! \brief Destructor.
     *
     * Unregisters the CUDA resource and deletes the OpenGL texture.
     */
    ~CudaTexture();

    /*! \brief Initializes the OpenGL texture and CUDA interoperability.
     *  \return True if initialization is successful, false otherwise.
     */
    bool init();

    /*! \brief Updates the texture using CUDA.
     *
     *  \param frame Pointer to the frame data.
     *  \param fd Reference to the frame descriptor.
     */
    void update(void* frame, const camera::FrameDescriptor& fd);

    /*! \brief Gets the OpenGL texture ID.
     *  \return OpenGL texture ID.
     */
    GLuint getTextureID() const { return Tex; }

    /*! \brief Gets the CUDA surface object.
     *  \return CUDA surface object.
     */
    cudaSurfaceObject_t getSurface() const { return cuSurface; }

  private:
    GLuint Tex;                       /*!< OpenGL texture ID. */
    cudaGraphicsResource* cuResource; /*!< CUDA resource for OpenGL/CUDA interoperability. */
    cudaSurfaceObject_t cuSurface;    /*!< CUDA surface object created from the texture. */
    cudaArray_t cuArray;              /*!< CUDA array mapped to the texture. */
    cudaStream_t cuStream;            /*!< CUDA stream for asynchronous operations. */

    int width;
    int height;
    camera::PixelDepth depth;

    CudaTexture(const CudaTexture&) = delete;
    CudaTexture& operator=(const CudaTexture&) = delete;
};

} // namespace holovibes::gui