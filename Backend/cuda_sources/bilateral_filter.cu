#include "bilateral_filter.cuh"
#include "cuda_memory.cuh" // Pour cudaSafeCall
#include <math.h>

/*!
 * \brief Kernel du filtre bilatéral.
 *
 * Pour chaque pixel, le kernel calcule une moyenne pondérée des pixels voisins,
 * avec des poids qui dépendent à la fois de la distance spatiale et de la différence d'intensité.
 */
__global__ void bilateral_filter_kernel(const float* input,
                                        float* output,
                                        const unsigned int width,
                                        const unsigned int height,
                                        const float sigma_spatial,
                                        const float sigma_range)
{
    // Calcul des indices de pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    // Valeur du pixel central
    float center_val = input[y * width + x];
    float sum = 0.0f;
    float norm = 0.0f;

    // Rayon du voisinage
    int radius = 9; // On peut ajuster ce rayon
    for (int j = -radius; j <= radius; j++)
    {
        for (int i = -radius; i <= radius; i++)
        {
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                float neighbor_val = input[ny * width + nx];
                // Poids spatial (basé sur la distance euclidienne)
                float spatial_weight = expf(-(i * i + j * j) / (2.0f * sigma_spatial * sigma_spatial));
                // Poids de différence d'intensité
                float range_weight = expf(-((neighbor_val - center_val) * (neighbor_val - center_val)) /
                                          (2.0f * sigma_range * sigma_range));
                float weight = spatial_weight * range_weight;
                sum += weight * neighbor_val;
                norm += weight;
            }
        }
    }
    // Valeur filtrée pour le pixel
    output[y * width + x] = sum / norm;
}

/*!
 * \brief Fonction hôte pour appliquer le filtre bilatéral sur l'image GPU.
 */
void apply_bilateral_filter(float* gpu_image,
                            const unsigned int width,
                            const unsigned int height,
                            const float sigma_spatial,
                            const float sigma_range,
                            const cudaStream_t stream)
{
    const unsigned int num_pixels = width * height;
    float* temp_buffer = nullptr;
    // Allocation d'un buffer temporaire pour stocker le résultat
    cudaSafeCall(cudaMalloc(&temp_buffer, num_pixels * sizeof(float)));

    // Configuration du kernel (taille des blocs et de la grille)
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Lancement du kernel du filtre bilatéral
    bilateral_filter_kernel<<<gridDim, blockDim, 0, stream>>>(gpu_image,
                                                              temp_buffer,
                                                              width,
                                                              height,
                                                              sigma_spatial,
                                                              sigma_range);

    // Copie asynchrone du résultat filtré dans le buffer d'origine
    cudaSafeCall(cudaMemcpyAsync(gpu_image, temp_buffer, num_pixels * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // Libération du buffer temporaire
    cudaSafeCall(cudaFree(temp_buffer));
}
