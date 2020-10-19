/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "tools_debug.hh"

#include <iostream>
#include <cufftXt.h>

#include "cuda_memory.cuh"

namespace holovibes
{
    void device_print(uchar* d_data, size_t offset, size_t nb_elts)
    {
        uchar* h_data = (uchar*)malloc(sizeof(uchar) * nb_elts);
        cudaXMemcpy(h_data, d_data + offset, sizeof(uchar) * nb_elts, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < nb_elts; ++i)
            std::cout << static_cast<uint>(h_data[i]) << std::endl;

        free(h_data);
    }

    void device_print(ushort* d_data, size_t offset, size_t nb_elts)
    {
        ushort* h_data = (ushort*)malloc(sizeof(ushort) * nb_elts);
        cudaXMemcpy(h_data, d_data + offset, sizeof(ushort) * nb_elts, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < nb_elts; ++i)
            std::cout << static_cast<uint>(h_data[i]) << std::endl;

        free(h_data);
    }

    void device_print(float* d_data, size_t offset, size_t nb_elts)
    {
        float* h_data = (float*)malloc(sizeof(float) * nb_elts);
        cudaXMemcpy(h_data, d_data + offset, sizeof(float) * nb_elts, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < nb_elts; ++i)
            std::cout << static_cast<uint>(h_data[i]) << std::endl;

        free(h_data);
    }

    void device_print(cuComplex* d_data, size_t offset, size_t nb_elts)
    {
        cuComplex* h_data = (cuComplex*)malloc(sizeof(cuComplex) * nb_elts);
        cudaXMemcpy(h_data, d_data + offset, sizeof(cuComplex) * nb_elts, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < nb_elts; ++i)
            std::cout << h_data[i].x << " " << h_data[i].y << "| " << std::endl;

        free(h_data);
    }
}