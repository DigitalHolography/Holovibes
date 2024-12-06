#include "globalpostprocess_api.hh"

namespace holovibes::api
{

#pragma region Internals

inline void GlobalPostProcessApi::set_convolution_enabled(bool value) { UPDATE_SETTING(ConvolutionEnabled, value); }

#pragma endregion

#pragma region Registration

void GlobalPostProcessApi::update_registration_zone(float value)
{
    if (!is_between(value, 0.f, 1.f) || api_->input.get_import_type() == ImportType::None)
        return;

    set_registration_zone(value);
    api_->compute.get_compute_pipe()->request(ICS::UpdateRegistrationZone);
}

void GlobalPostProcessApi::set_registration_enabled(bool value)
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    UPDATE_SETTING(RegistrationEnabled, value);
    api_->compute.get_compute_pipe()->request(ICS::UpdateRegistrationZone);
}

#pragma endregion

#pragma region Renormalization

void GlobalPostProcessApi::set_renorm_enabled(bool value)
{
    UPDATE_SETTING(RenormEnabled, value);
    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Conv Matrix

static inline const std::filesystem::path dir(GET_EXE_DIR);

/*!
 * \brief Loads a convolution matrix from a file
 *
 * This function is a tool / util supposed to be called by other functions
 *
 * \param[in] file The name of the file to load the matrix from. NOT A FULL PATH
 * \param[in] convo_matrix Where to store the read matrix
 *
 * \throw std::runtime_error runtime_error When the matrix cannot be loaded
 */
void GlobalPostProcessApi::load_convolution_matrix_file(const std::string& file, std::vector<float>& convo_matrix)
{
    auto& holo = Holovibes::instance();

    auto path_file = dir / __CONVOLUTION_KERNEL_FOLDER_PATH__ / file;
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

void GlobalPostProcessApi::load_convolution_matrix(std::string filename)
{
    set_convolution_enabled(true);
    set_convo_matrix({});

    // There is no file None.txt for convolution
    if (filename.empty())
        return;

    std::vector<float> convo_matrix = get_convo_matrix();

    try
    {
        load_convolution_matrix_file(filename, convo_matrix);
        set_convo_matrix(convo_matrix);
    }
    catch (std::exception& e)
    {
        set_convo_matrix({});
        LOG_ERROR("Couldn't load convolution matrix : {}", e.what());
    }
}

#pragma endregion

#pragma region Conv Divide

void GlobalPostProcessApi::set_divide_convolution_enabled(const bool value)
{
    if (api_->input.get_import_type() == ImportType::None || get_divide_convolution_enabled() == value ||
        !get_convolution_enabled())
        return;

    UPDATE_SETTING(DivideConvolutionEnabled, value);
    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Convolution

void GlobalPostProcessApi::enable_convolution(const std::string& filename)
{
    if (api_->input.get_import_type() == ImportType::None)
        return;

    set_convolution_file_name(filename);
    load_convolution_matrix(filename);

    if (filename.empty())
    {
        api_->compute.pipe_refresh();
        return;
    }

    try
    {
        auto pipe = api_->compute.get_compute_pipe();
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

void GlobalPostProcessApi::disable_convolution()
{
    set_convo_matrix({});
    set_convolution_enabled(false);
    try
    {
        auto pipe = api_->compute.get_compute_pipe();
        pipe->request(ICS::DisableConvolution);
        while (pipe->is_requested(ICS::DisableConvolution))
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }
}

#pragma endregion

} // namespace holovibes::api