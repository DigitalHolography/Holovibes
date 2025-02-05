#include "vessel_enhancement.cuh"
#include "cuda_memory.cuh" // Pour cudaSafeCall, par exemple
#include <math.h>

/*!
 * \brief Kernel CUDA pour le calcul de la mesure vesselness.
 *
 * Pour chaque pixel (à l'intérieur des limites pour le calcul des différences), on calcule
 * les dérivées secondes approximatives Hxx, Hyy et Hxy via différences centrales.
 * On en déduit ensuite les valeurs propres de la matrice Hessienne 2x2.
 * Si la plus grande valeur propre (en valeur absolue) est négative, on calcule Ra et S,
 * et la vesselness selon la formule de Frangi.
 *
 * \param[in] input   Image d'entrée (format float).
 * \param[out] output Image de sortie (mesure vesselness).
 * \param[in] width   Largeur de l'image.
 * \param[in] height  Hauteur de l'image.
 * \param[in] beta    Paramètre beta.
 * \param[in] c       Paramètre c.
 */
__global__ void vesselness_filter_kernel(const float* input, float* output, int width, int height, float beta, float c)
{
    // Calcul des coordonnées du pixel courant
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Pour simplifier, on ne traite pas les bords
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    {
        if (x < width && y < height)
            output[y * width + x] = 0.0f;
        return;
    }

    // Lecture de la valeur centrale
    float center = input[y * width + x];

    // Calcul des dérivées secondes par différences centrales
    float Hxx = input[y * width + (x + 1)] - 2.0f * center + input[y * width + (x - 1)];
    float Hyy = input[(y + 1) * width + x] - 2.0f * center + input[(y - 1) * width + x];
    float Hxy = (input[(y + 1) * width + (x + 1)] - input[(y + 1) * width + (x - 1)] -
                 input[(y - 1) * width + (x + 1)] + input[(y - 1) * width + (x - 1)]) /
                4.0f;

    // Calcul de la trace et de la différence
    float trace = Hxx + Hyy;
    float diff = Hxx - Hyy;
    // Calcul du discriminant
    float discrim = sqrtf(0.25f * diff * diff + Hxy * Hxy);
    float lambda1 = 0.5f * trace + discrim;
    float lambda2 = 0.5f * trace - discrim;

    // On souhaite avoir |lambda1| <= |lambda2|, sinon on échange
    if (fabsf(lambda1) > fabsf(lambda2))
    {
        float tmp = lambda1;
        lambda1 = lambda2;
        lambda2 = tmp;
    }

    // Calcul de la vesselness : si lambda2 n'est pas négative, le pixel n'est pas considéré comme appartenant à un
    // vaisseau.
    float vesselness = 0.0f;
    if (lambda2 < 0)
    {
        float Ra = fabsf(lambda1) / fabsf(lambda2);
        float S = sqrtf(lambda1 * lambda1 + lambda2 * lambda2);
        float expRa = expf(-(Ra * Ra) / (2.0f * beta * beta));
        float expS = 1.0f - expf(-(S * S) / (2.0f * c * c));
        vesselness = expRa * expS;
    }

    output[y * width + x] = vesselness;
}

/*!
 * \brief Fonction hôte qui applique le filtre vesselness sur une image GPU.
 *
 * La fonction alloue un buffer temporaire pour le résultat intermédiaire,
 * lance le kernel pour calculer la mesure vesselness, copie le résultat dans l'image d'entrée,
 * et libère le buffer temporaire.
 *
 * \param[in,out] gpu_image Pointeur sur l'image GPU (format float) à traiter.
 * \param[in] width         Largeur de l'image.
 * \param[in] height        Hauteur de l'image.
 * \param[in] beta          Paramètre beta (ex. 0.5f).
 * \param[in] c             Paramètre c (ex. 15.0f).
 * \param[in] stream        Stream CUDA à utiliser.
 */
void apply_vesselness_filter(
    float* gpu_image, unsigned int width, unsigned int height, float beta, float c, cudaStream_t stream)
{
    const unsigned int num_pixels = width * height;
    float* temp_buffer = nullptr;
    cudaSafeCall(cudaMalloc(&temp_buffer, num_pixels * sizeof(float)));

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    vesselness_filter_kernel<<<gridDim, blockDim, 0, stream>>>(gpu_image, temp_buffer, width, height, beta, c);

    cudaSafeCall(cudaMemcpyAsync(gpu_image, temp_buffer, num_pixels * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    cudaSafeCall(cudaFree(temp_buffer));

    // Optionnel : synchronisation du stream si nécessaire
    // cudaSafeCall(cudaStreamSynchronize(stream));
}
