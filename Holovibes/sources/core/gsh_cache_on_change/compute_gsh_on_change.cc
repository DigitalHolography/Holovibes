#include "compute_gsh_on_change.hh"
#include "common_gsh_on_change.hh"
#include "API.hh"

namespace holovibes
{

static void load_convolution_matrix(ConvolutionStruct& convo);
template <>
void ComputeGSHOnChange::operator()<Convolution>(ConvolutionStruct& new_value)
{
    LOG_UPDATE_ON_CHANGE(Convolution);

    load_convolution_matrix(new_value);
}

static void load_convolution_matrix(ConvolutionStruct& convo)
{
    if (api::get_convolution().enabled == false || api::get_convolution().type == UID_CONVOLUTION_TYPE_DEFAULT)
        return;

    std::filesystem::path dir(get_exe_dir());
    dir = dir / "convolution_kernels" / convo.type;
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
    const uint output_width = api::get_output_frame_descriptor().width;
    const uint output_height = api::get_output_frame_descriptor().height;
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

    convo.matrix.clear();
    convo.matrix.resize(size, 0.0f);

    uint kernel_indice = 0;
    for (uint i = first_row; i < last_row; i++)
    {
        for (uint j = first_col; j < last_col; j++)
        {
            convo.matrix[i * output_width + j] = matrix[kernel_indice];
            kernel_indice++;
        }
    }
}

template <>
void ComputeGSHOnChange::operator()<BatchSize>(int& new_value)
{
    LOG_UPDATE_ON_CHANGE(BatchSize);

    if (new_value > api::get_input_buffer_size())
        new_value = api::get_input_buffer_size();

    auto time_stride = api::get_time_stride();
    if (time_stride < new_value)
        api::set_time_stride(new_value);
    else if (time_stride % new_value != 0)
        api::set_time_stride(time_stride - time_stride % new_value);
}

template <>
void ComputeGSHOnChange::operator()<TimeStride>(int& new_value)
{
    LOG_UPDATE_ON_CHANGE(TimeStride);

    auto batch_size = api::get_batch_size();
    if (batch_size > new_value)
        new_value = batch_size;
    else if (new_value % batch_size != 0)
        new_value = new_value - new_value % batch_size;
}

template <>
void ComputeGSHOnChange::operator()<TimeTransformationCutsEnable>(bool& new_value)
{
    LOG_UPDATE_ON_CHANGE(TimeTransformationCutsEnable);

    if (new_value == false)
        api::detail::set_value<CutsViewEnabled>(false);
}

template <>
void ComputeGSHOnChange::operator()<ComputeMode>(ComputeModeEnum& new_value)
{
    LOG_UPDATE_ON_CHANGE(ComputeMode);

    compute_output_fd(api::get_import_frame_descriptor(), new_value, api::get_image_type());
}

template <>
void ComputeGSHOnChange::operator()<ImageType>(ImageTypeEnum& new_value)
{
    LOG_UPDATE_ON_CHANGE(ImageType);

    compute_output_fd(api::get_import_frame_descriptor(), api::get_compute_mode(), new_value);
}

template <>
void ComputeGSHOnChange::operator()<Filter2D>(Filter2DStruct& new_value)
{
    LOG_UPDATE_ON_CHANGE(Filter2D);

    if (new_value.enabled == false)
        api::set_filter2d_view_enabled(false);
}

template <>
void ComputeGSHOnChange::operator()<SpaceTransformation>(SpaceTransformationEnum& new_value)
{
    if (new_value == SpaceTransformationEnum::FFT1)
        api::detail::set_value<FftShiftEnabled>(true);
    else
        api::detail::set_value<FftShiftEnabled>(false);
}
} // namespace holovibes