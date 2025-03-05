
#ifdef WIN32
#include <windows.h>
#endif
#include <cuda_gl_interop.h>
#include "CudaTexture.hh"
#include "texture_update.cuh"

namespace holovibes::gui
{

CudaTexture::CudaTexture(int width, int height, camera::PixelDepth depth, cudaStream_t stream)
    : Tex(0)
    , cuResource(nullptr)
    , cuSurface(0)
    , cuArray(nullptr)
    , cuStream(stream)
    , width(width)
    , height(height)
    , depth(depth)
{
}

CudaTexture::~CudaTexture()
{
    // Destroy the CUDA surface object if it was created.
    if (cuSurface)
    {
        cudaDestroySurfaceObject(cuSurface);
    }
    // Unregister the CUDA resource if it was registered.
    if (cuResource)
    {
        cudaGraphicsUnregisterResource(cuResource);
    }
    // Delete the OpenGL texture.
    if (Tex)
    {
        glDeleteTextures(1, &Tex);
    }
}

bool CudaTexture::init()
{
    // Generate and bind the OpenGL texture.
    glGenTextures(1, &Tex);
    glBindTexture(GL_TEXTURE_2D, Tex);

    // Create an empty image to initialize the texture.
    // We use an array of unsigned char (1 byte per pixel) for GL_RED format.
    size_t size = width * height;
    unsigned char* mTexture8 = new unsigned char[size];
    std::memset(mTexture8, 0, size * sizeof(unsigned char));

    // Initialize the texture with the empty image.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, mTexture8);
    delete[] mTexture8;

    // Set texture parameters.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    // Configure swizzling based on the pixel type.
    if (depth == camera::PixelDepth::Complex)
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_ZERO);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_GREEN);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
    }

    // Unbind the texture.
    glBindTexture(GL_TEXTURE_2D, 0);

    // Register the OpenGL texture with CUDA for interoperability.
    cudaError_t err =
        cudaGraphicsGLRegisterImage(&cuResource, Tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (err != cudaSuccess)
    {
        return false;
    }

    // Map the CUDA resource and obtain the associated array.
    cudaGraphicsMapResources(1, &cuResource, cuStream);
    cudaGraphicsSubResourceGetMappedArray(&cuArray, cuResource, 0, 0);

    // Prepare the CUDA resource descriptor.
    cudaResourceDesc cuResDesc;
    std::memset(&cuResDesc, 0, sizeof(cuResDesc));
    cuResDesc.resType = cudaResourceTypeArray;
    cuResDesc.res.array.array = cuArray;

    // Create the CUDA surface object from the array.
    err = cudaCreateSurfaceObject(&cuSurface, &cuResDesc);
    if (err != cudaSuccess)
    {
        return false;
    }

    return true;
}

void CudaTexture::update(void* frame, const camera::FrameDescriptor& fd)
{
    // Mapper la ressource CUDA pour accéder au tableau associé à la texture.
    cudaGraphicsMapResources(1, &cuResource, cuStream);

    // Récupérer le tableau mappé.
    cudaGraphicsSubResourceGetMappedArray(&cuArray, cuResource, 0, 0);

    // (Optionnel) Si nécessaire, recréer le cudaSurfaceObject pour prendre en compte le tableau mappé.
    cudaResourceDesc resDesc;
    std::memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    // Détruire l'ancien objet surface s'il existe
    if (cuSurface)
    {
        cudaDestroySurfaceObject(cuSurface);
    }
    cudaCreateSurfaceObject(&cuSurface, &resDesc);

    // Lancer le kernel CUDA pour mettre à jour la texture avec les données du frame.
    textureUpdate(cuSurface, frame, fd, cuStream);

    // Démapper la ressource pour permettre à OpenGL d'accéder aux données mises à jour.
    cudaGraphicsUnmapResources(1, &cuResource, cuStream);
}

} // namespace holovibes::gui