#include "globalpostprocess_api.hh"

#include "API.hh"

#define NOT_SAME_AND_NOT_RAW(old_val, new_val)                                                                         \
    if (old_val == new_val)                                                                                            \
        return ApiCode::NO_CHANGE;                                                                                     \
    if (api_->compute.get_compute_mode() == Computation::Raw)                                                          \
        return ApiCode::WRONG_COMP_MODE;

namespace holovibes::api
{

#pragma region Registration

ApiCode GlobalPostProcessApi::set_registration_enabled(bool value) const
{
    NOT_SAME_AND_NOT_RAW(get_registration_enabled(), value);

    UPDATE_SETTING(RegistrationEnabled, value);

    if (!api_->compute.get_is_computation_stopped())
        api_->compute.get_compute_pipe()->request(ICS::UpdateRegistrationZone);

    return ApiCode::OK;
}

ApiCode GlobalPostProcessApi::set_registration_zone(float value) const
{
    NOT_SAME_AND_NOT_RAW(get_registration_zone(), value);

    if (!get_registration_enabled())
        return ApiCode::INVALID_VALUE;

    if (!is_between(value, 0.f, 1.f))
    {
        LOG_WARN("Registration zone must be in range ]0, 1[");
        return ApiCode::INVALID_VALUE;
    }

    UPDATE_SETTING(RegistrationZone, value);

    if (!api_->compute.get_is_computation_stopped())
        api_->compute.get_compute_pipe()->request(ICS::UpdateRegistrationZone);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Renormalization

ApiCode GlobalPostProcessApi::set_renorm_enabled(bool value) const
{
    NOT_SAME_AND_NOT_RAW(get_renorm_enabled(), value);

    UPDATE_SETTING(RenormEnabled, value);

    return ApiCode::OK;
}

ApiCode GlobalPostProcessApi::set_renorm_constant(unsigned int value) const
{
    if (get_renorm_constant() == value)
        return ApiCode::NO_CHANGE;

    UPDATE_SETTING(RenormConstant, value);

    return ApiCode::OK;
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

    std::vector<float> convo_matrix = {};
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

ApiCode GlobalPostProcessApi::set_divide_convolution_enabled(const bool value) const
{
    NOT_SAME_AND_NOT_RAW(get_divide_convolution_enabled(), value);

    if (get_convolution_file_name().empty())
        return ApiCode::INVALID_VALUE;

    UPDATE_SETTING(DivideConvolutionEnabled, value);

    return ApiCode::OK;
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

    std::vector<float> convo_matrix = load_convolution_matrix(filename);
    UPDATE_SETTING(ConvolutionMatrix, convo_matrix);

    auto request = convo_matrix.empty() ? ICS::DisableConvolution : ICS::Convolution;
    api_->compute.get_compute_pipe()->request(request);

    return ApiCode::OK;
}

#pragma endregion

} // namespace holovibes::api