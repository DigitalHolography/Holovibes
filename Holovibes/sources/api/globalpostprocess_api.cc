#include "globalpostprocess_api.hh"

namespace holovibes::api
{

#pragma region Registration

void update_registration_zone(float value)
{
    if (!is_between(value, 0.f, 1.f) || api::get_import_type() == ImportType::None)
        return;

    set_registration_zone(value);
    api::get_compute_pipe()->request(ICS::UpdateRegistrationZone);
    pipe_refresh();
}

#pragma endregion

#pragma region Convolution

static inline const std::filesystem::path dir(GET_EXE_DIR);

/**
 * \brief Loads a convolution matrix from a file
 *
 * This function is a tool / util supposed to be called by other functions
 *
 * \param file The name of the file to load the matrix from. NOT A FULL PATH
 * \param convo_matrix Where to store the read matrix
 *
 * \throw std::runtime_error runtime_error When the matrix cannot be loaded
 */
void load_convolution_matrix_file(const std::string& file, std::vector<float>& convo_matrix)
{
    auto& holo = Holovibes::instance();

    auto path_file = dir / __CONVOLUTION_KERNEL_FOLDER_PATH__ / file; //"convolution_kernels" / file;
    std::string path = path_file.string();

    std::vector<float> matrix;
    uint matrix_width = 0;
    uint matrix_height = 0;
    uint matrix_z = 1;

    // Doing this the C way because it's faster
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

    convo_matrix.resize(size, 0.0f);

    uint kernel_indice = 0;
    for (uint i = first_row; i < last_row; i++)
    {
        for (uint j = first_col; j < last_col; j++)
        {
            (convo_matrix)[i * output_width + j] = matrix[kernel_indice];
            kernel_indice++;
        }
    }
}

void load_convolution_matrix(std::string filename)
{
    api::set_convolution_enabled(true);
    api::set_convo_matrix({});

    // There is no file None.txt for convolution
    if (filename.empty())
        return;

    std::vector<float> convo_matrix = api::get_convo_matrix();

    try
    {
        load_convolution_matrix_file(filename, convo_matrix);
        api::set_convo_matrix(convo_matrix);
    }
    catch (std::exception& e)
    {
        api::set_convo_matrix({});
        LOG_ERROR("Couldn't load convolution matrix : {}", e.what());
    }
}

void enable_convolution(const std::string& filename)
{
    if (api::get_import_type() == ImportType::None)
        return;

    api::set_convolution_file_name(filename);

    load_convolution_matrix(filename);

    if (filename.empty())
    {
        pipe_refresh();
        return;
    }

    try
    {
        auto pipe = get_compute_pipe();
        pipe->request(ICS::Convolution);
        // Wait for the convolution to be enabled for notify
        while (pipe->is_requested(ICS::Convolution))
            continue;
    }
    catch (const std::exception& e)
    {
        disable_convolution();
        LOG_ERROR("Catch {}", e.what());
    }
}

void disable_convolution()
{
    set_convo_matrix({});
    set_convolution_enabled(false);
    try
    {
        auto pipe = get_compute_pipe();
        pipe->request(ICS::DisableConvolution);
        while (pipe->is_requested(ICS::DisableConvolution))
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }
}

void set_divide_convolution(const bool value)
{
    if (get_import_type() == ImportType::None || get_divide_convolution_enabled() == value ||
        !get_convolution_enabled())
        return;

    set_divide_convolution_enabled(value);
    pipe_refresh();
}

#pragma endregion

} // namespace holovibes::api