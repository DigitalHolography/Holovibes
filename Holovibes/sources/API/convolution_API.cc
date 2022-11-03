#include "API.hh"

namespace holovibes::api
{

void enable_convolution(const std::string& filename)
{
    {
        api::change_convolution()->set_is_enabled(true);
        api::change_convolution()->get_matrix_ref().clear();

        if (filename != UID_CONVOLUTION_TYPE_DEFAULT)
            load_convolution_matrix(filename);
    }

    while (api::get_compute_pipe().get_composite_cache().has_change_requested())
        continue;
}

void disable_convolution()
{
    {
        api::change_convolution()->set_is_enabled(false);
        api::change_convolution()->get_matrix_ref().clear();
    }

    while (api::get_compute_pipe().get_composite_cache().has_change_requested())
        continue;
}

void load_convolution_matrix(const std::string& file)
{
    auto& holo = Holovibes::instance();

    try
    {
        std::filesystem::path dir(get_exe_dir());
        dir = dir / "convolution_kernels" / file;
        std::string path = dir.string();

        std::vector<float> matrix;
        uint matrix_width = 0;
        uint matrix_height = 0;
        uint matrix_z = 1;

        // Doing this the C way cause it's faster
        FILE* c_file;
        fopen_s(&c_file, path.c_str(), "r");

        if (c_file == nullptr)
        {
            fclose(c_file);
            throw std::runtime_error("Invalid file path");
        }

        // Read kernel dimensions
        if (fscanf_s(c_file, "%u %u %u;", &matrix_width, &matrix_height, &matrix_z) != 3)
        {
            fclose(c_file);
            throw std::runtime_error("Invalid kernel dimensions");
        }

        size_t matrix_size = matrix_width * matrix_height * matrix_z;
        matrix.resize(matrix_size);

        // Read kernel values
        for (size_t i = 0; i < matrix_size; ++i)
        {
            if (fscanf_s(c_file, "%f", &matrix[i]) != 1)
            {
                fclose(c_file);
                throw std::runtime_error("Missing values");
            }
        }

        fclose(c_file);

        // Reshape the vector as a (nx,ny) rectangle, keeping z depth
        const uint output_width = holo.get_gpu_output_queue()->get_fd().width;
        const uint output_height = holo.get_gpu_output_queue()->get_fd().height;
        const uint size = output_width * output_height;

        // The convo matrix is centered and padded with 0 since the kernel is
        // usally smaller than the output Example: kernel size is (2, 2) and
        // output size is (4, 4) The kernel is represented by 'x' and
        //  | 0 | 0 | 0 | 0 |
        //  | 0 | x | x | 0 |
        //  | 0 | x | x | 0 |
        //  | 0 | 0 | 0 | 0 |
        const uint first_col = (output_width / 2) - (matrix_width / 2);
        const uint last_col = (output_width / 2) + (matrix_width / 2);
        const uint first_row = (output_height / 2) - (matrix_height / 2);
        const uint last_row = (output_height / 2) + (matrix_height / 2);

        GSH::instance().get_compute_cache().get_value_ref_W<Convolution_PARAM>().get_matrix_ref().resize(size, 0.0f);

        uint kernel_indice = 0;
        for (uint i = first_row; i < last_row; i++)
        {
            for (uint j = first_col; j < last_col; j++)
            {
                GSH::instance()
                    .get_compute_cache()
                    .get_value_ref_W<Convolution_PARAM>()
                    .get_matrix_ref()[i * output_width + j] = matrix[kernel_indice];
                kernel_indice++;
            }
        }
    }
    catch (std::exception& e)
    {
        GSH::instance().get_compute_cache().get_value_ref_W<Convolution_PARAM>().get_matrix_ref().clear();
        LOG_ERROR(main, "Couldn't load convolution matrix : {}", e.what());
    }
    GSH::instance().get_compute_cache().force_trigger_param_W<Convolution_PARAM>();
}

} // namespace holovibes::api
