#include "cuda_memory.cuh"
#include "common.cuh"
#include "tools_analysis.cuh"
#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "cuComplex.h"
#include "cufft_handle.hh"
#include "cusolverDn.h"

namespace
{
template <typename T>
__global__ void kernel_multiply_array_by_scalar(T* input_output, size_t size, const T scalar)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        input_output[index] *= scalar;
    }
}

template <typename T>
void multiply_array_by_scalar_caller(T* input_output, size_t size, T scalar, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_array_by_scalar<<<blocks, threads, 0, stream>>>(input_output, size, scalar);
}
}

void multiply_array_by_scalar(float* input_output, size_t size, float scalar, cudaStream_t stream)
{
    multiply_array_by_scalar_caller<float>(input_output, size, scalar, stream);
}

// CUDA kernel to prepare H hessian matrices
__global__ void prepareHessian(float* output, const float* ixx, const float* ixy, const float* iyx, const float* iyy, const int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        // Prepare the 2x2 submatrix for point `index`
        output[index * 4 + 0] = ixx[index];
        output[index * 4 + 1] = ixy[index];
        output[index * 4 + 2] = iyx[index];
        output[index * 4 + 3] = iyy[index];
    }
}

void compute_sorted_eigenvalues_2x2(float* H, int frame_res, float* lambda1, float* lambda2, cudaStream_t stream)
{
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);
    cusolverDnSetStream(cusolver_handle, stream);

    int lwork; // Taille du tampon de travail
    float* work_buffer; // Tampon de travail sur le GPU
    int* dev_info; // Statut d'erreur pour chaque appel de cusolver
    cudaXMalloc(&dev_info, sizeof(int));
    int h_meig; // Nombre de valeurs propres trouvées
    float *hessian_2x2 = H;

    // Calculer la taille nécessaire du tampon de travail
    cusolverStatus_t status_buffer = cusolverDnSsyevdx_bufferSize(
            cusolver_handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUSOLVER_EIG_RANGE_ALL,
            CUBLAS_FILL_MODE_LOWER,
            2,                       // Dimension de la matrice
            hessian_2x2,             // La matrice d'entrée
            2,                       // Leading dimension de la matrice
            -FLT_MAX, FLT_MAX,       // Plage complète de valeurs propres
            1, 2,                    // Indices pour toute la plage
            &h_meig,                 // Nombre de valeurs propres trouvées (sortie)
            nullptr,                 // Vecteur des valeurs propres (non utilisé ici)
            &lwork                   // Retourne la taille du tampon de travail ici
        );

    // Allouer le tampon de travail sur le GPU
    cudaXMalloc((void**)&work_buffer, lwork * sizeof(float));

    float* eigenvalues; // Pour stocker les valeurs propres
    cudaXMalloc(&eigenvalues, sizeof(float) * 2);
    for (int i = 0; i < frame_res; ++i)
    {
        // Préparer la sous-matrice 2x2 pour le point `i`

        hessian_2x2 = H + (4 * i);

        if (status_buffer != CUSOLVER_STATUS_SUCCESS)
        {
            std::cerr << "Erreur lors de la récupération de la taille du tampon de travail pour le point " << i << std::endl;
            continue;
        }

        cusolverStatus_t status = cusolverDnSsyevdx(
            cusolver_handle,
            CUSOLVER_EIG_MODE_VECTOR,
            CUSOLVER_EIG_RANGE_ALL,
            CUBLAS_FILL_MODE_LOWER,
            2,
            hessian_2x2,               // Matrice d'entrée
            2,                          // Leading dimension
            -FLT_MAX, FLT_MAX,         // Plage complète de valeurs propres
            1, 2,                       // Indices pour toute la plage
            &h_meig,                    // Nombre de valeurs propres trouvées
            eigenvalues,                // Vecteur pour stocker les valeurs propres
            work_buffer,                // Tampon de travail
            lwork,                      // Taille du tampon
            dev_info                    // Statut d'erreur
        );

        // Libérer le tampon de travail pour ce point
    }

    std::cout << "fini" << std::endl;

    cudaXFree(eigenvalues);
    cudaXFree(work_buffer);
    cudaXFree(dev_info);
    // Détruire le handle cusolver
    cusolverDnDestroy(cusolver_handle);
}

// Fonction pour calculer les valeurs propres d'une matrice 2x2
void calculerValeursPropres(float a, float b, float d, float* lambda1, float* lambda2, int ind)
{
    // Calcul de la trace et du déterminant
    float trace = a + d;
    float determinant = a * d - b * b;

    // Calcul du discriminant
    float discriminant = trace * trace - 4 * determinant;

    // Vérification si les valeurs propres sont réelles
    if (discriminant >= 0) {
        float eig1 = (trace + std::sqrt(discriminant)) / 2;
        float eig2 = (trace - std::sqrt(discriminant)) / 2;
        // Vérification de la condition
        if (std::abs(eig1) < std::abs(eig2)) {
            lambda1[ind] = eig1;
            lambda2[ind] = eig2;
        } else {
            lambda1[ind] = eig2;
            lambda2[ind] = eig1;
        }
    } else {
        std::cout << "Les valeurs propres ne sont pas réelles." << std::endl;
    }
}

__global__ void kernel_padding(float* output, float* input, int height, int width, int new_width, int start_x, int start_y) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int y = idx / width;
    int x = idx % width;

    if (y < height && x < width) 
    {
        output[(start_y + y) * new_width + (start_x + x)] = input[y * width + x];
    }
}

// Useless
void convolution_kernel_add_padding(float* output, float* kernel, const int width, const int height, const int new_width, const int new_height, cudaStream_t stream) 
{
    //float* padded_kernel = new float[new_width * new_height];
    //std::memset(padded_kernel, 0, new_width * new_height * sizeof(float));

    int start_x = (new_width - width) / 2;
    int start_y = (new_height - height) / 2;

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(width * height, threads);
    kernel_padding<<<blocks, threads, 0, stream>>>(output, kernel, height, width, new_width, start_x, start_y);

}

void write1DFloatArrayToFile(const float* array, int rows, int cols, const std::string& filename)
{
    // Open the file in write mode
    std::ofstream outFile(filename);

    // Check if the file was opened successfully
    if (!outFile)
    {
        std::cerr << "Error: Unable to open the file " << filename << std::endl;
        return;
    }

    // Write the 1D array in row-major order to the file
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outFile << array[i * cols + j]; // Calculate index in row-major order
            if (j < cols - 1)
            {
                outFile << " "; // Separate values in a row by a space
            }
        }
        outFile << std::endl; // New line after each row
    }

    // Close the file
    outFile.close();
    std::cout << "1D array written to the file " << filename << std::endl;
}

void print_in_file(float* input, uint size, std::string filename, cudaStream_t stream)
{
    float* result = new float[size];
    cudaXMemcpyAsync(result,
                        input,
                        size * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream);
    write1DFloatArrayToFile(result,
                            sqrt(size),
                            sqrt(size),
                            "test_" + filename + ".txt");
}