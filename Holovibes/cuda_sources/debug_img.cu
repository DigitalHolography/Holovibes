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

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <memory>

void from_gpu_img_to_csv(const float *gpu_img, const size_t frame_res, const std::string output_path, const size_t img_nb)
{
	float* local_img = new float[frame_res];
	

	std::ofstream file;
	file.open(output_path);
	
	
	for (size_t i = 0; i < img_nb; i++)
	{
		cudaMemcpy(local_img, gpu_img + (frame_res * i), frame_res * sizeof(float), cudaMemcpyDeviceToHost);
		for (size_t j = 0; j < frame_res; j++)
		{
			file << local_img[j];
			if (j < frame_res - 1)
				file << ',';
		}
		file << std::endl;
	}

	file.close();
}