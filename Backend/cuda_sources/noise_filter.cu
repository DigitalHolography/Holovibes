#include "noise_filter.cuh"
#include "cuda_memory.cuh" // Pour cudaSafeCall, par exemple
#include "logger.hh"       // Pour les logs si besoin

/*!
 * \brief Kernel CUDA qui applique un filtre moyenneur 3x3 sur une image.
 *
 * \param[in] input  Image d'entrée en format float.
 * \param[out] output Image de sortie en format float.
 * \param[in] width  Largeur de l'image.
 * \param[in] height Hauteur de l'image.
 */
__global__ void noise_filter_kernel(const float* input, float* output, int width, int height)
{
    // Calcul des coordonnées du pixel courant
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float sum = 0.0f;
    int count = 0;

    // Parcours du voisinage 3x3
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;
            // Vérifier que le voisin est dans l'image
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                sum += input[ny * width + nx];
                ++count;
            }
        }
    }
    // Calcul de la moyenne et écriture du résultat
    output[y * width + x] = sum / count;
}

/*!
 * \brief Applique le filtre anti bruit sur l'image passée en paramètre.
 *
 * \param[in,out] gpu_image Pointeur sur l'image GPU (format float) à filtrer.
 * \param[in] width  Largeur de l'image.
 * \param[in] height Hauteur de l'image.
 * \param[in] stream Stream CUDA à utiliser pour l'exécution.
 */
void apply_noise_filter(float* gpu_image, const uint width, const uint height, const cudaStream_t stream)
{
    // Calcul du nombre total de pixels
    const uint num_pixels = width * height;
    float* temp_buffer = nullptr;

    // Allocation d'un buffer temporaire sur le GPU pour stocker l'image filtrée
    cudaSafeCall(cudaMalloc(&temp_buffer, num_pixels * sizeof(float)));

    // Définition de la configuration d'exécution (taille des blocs et de la grille)
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Lancement du kernel de filtrage
    noise_filter_kernel<<<gridDim, blockDim, 0, stream>>>(gpu_image, temp_buffer, width, height);

    // Copie asynchrone du résultat filtré dans le buffer d'origine
    cudaSafeCall(cudaMemcpyAsync(gpu_image, temp_buffer, num_pixels * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // Libération du buffer temporaire
    cudaSafeCall(cudaFree(temp_buffer));

    // (Optionnel) Synchroniser le stream si nécessaire
    cudaSafeCall(cudaStreamSynchronize(stream));
}
