#include "globalpostprocess_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Registration

void GlobalPostProcessApi::update_registration_zone(float value) const
{
    if (!is_between(value, 0.f, 1.f) || api_->compute.get_is_computation_stopped())
        return;

    set_registration_zone(value);
    api_->compute.get_compute_pipe()->request(ICS::UpdateRegistrationZone);
}

void GlobalPostProcessApi::set_registration_enabled(bool value) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    UPDATE_SETTING(RegistrationEnabled, value);
    api_->compute.get_compute_pipe()->request(ICS::UpdateRegistrationZone);
}

#pragma endregion

#pragma region Renormalization

void GlobalPostProcessApi::set_renorm_enabled(bool value) const
{
    UPDATE_SETTING(RenormEnabled, value);
    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Conv Matrix

static inline const std::filesystem::path dir(GET_EXE_DIR);

std::vector<float> GlobalPostProcessApi::load_convolution_matrix(const std::string& file) const
{
    if (file.empty())
        return {};

    auto path_file = dir / __CONVOLUTION_KERNEL_FOLDER_PATH__ / file;
    if (!std::filesystem::exists(path_file))
    {
        LOG_WARN("[Convolution: File not found : {}. Convolution deactivated", path_file.string());
        return {};
    }

    std::string path = path_file.string();

    std::vector<float> convo_matrix = get_convo_matrix();
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
        LOG_ERROR("[Convolution]: Couldn't open file {}", path);
        return {};
    }

    // Read kernel dimensions
    if (fscanf_s(c_file, "%u %u %u;", &matrix_width, &matrix_height, &matrix_z) != 3)
    {
        fclose(c_file);
        LOG_ERROR("[Convolution]: Invalid kernel dimensions");
        return {};
    }

    size_t matrix_size = matrix_width * matrix_height * matrix_z;
    matrix.resize(matrix_size);

    // Read kernel values
    for (size_t i = 0; i < matrix_size; ++i)
    {
        if (fscanf_s(c_file, "%f", &matrix[i]) != 1)
        {
            fclose(c_file);
            LOG_ERROR("[Convolution]: Missing values in kernel");
            return {};
        }
    }

    fclose(c_file);

    // Reshape the vector as a (nx,ny) rectangle, keeping z depth
    const uint output_width = api_->compute.get_gpu_output_queue()->get_fd().width;
    const uint output_height = api_->compute.get_gpu_output_queue()->get_fd().height;
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

    return convo_matrix;
}

#pragma endregion

#pragma region Conv Divide

void GlobalPostProcessApi::set_divide_convolution_enabled(const bool value) const
{
    if (api_->compute.get_is_computation_stopped() || get_divide_convolution_enabled() == value ||
        get_convolution_file_name().empty())
        return;

    UPDATE_SETTING(DivideConvolutionEnabled, value);
    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Convolution

ApiCode GlobalPostProcessApi::enable_convolution(const std::string& filename) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return ApiCode::WRONG_COMP_MODE;

    UPDATE_SETTING(ConvolutionFileName, filename);

    if (api_->compute.get_is_computation_stopped())
        return ApiCode::OK;

    UPDATE_SETTING(ConvolutionMatrix, {});
    std::vector<float> convo_matrix = load_convolution_matrix(filename);
    UPDATE_SETTING(ConvolutionMatrix, convo_matrix);

    auto request = convo_matrix.empty() ? ICS::DisableConvolution : ICS::Convolution;
    auto pipe = api_->compute.get_compute_pipe();
    pipe->request(request);
    while (pipe->is_requested(request))
        continue;

    return ApiCode::OK;
}

#pragma endregion

} // namespace holovibes::api