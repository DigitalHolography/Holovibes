#include "compute_gsh_on_change.hh"
#include "API.hh"

namespace holovibes
{

static void load_convolution_matrix(ConvolutionStruct& convo);
template <>
void ComputeGSHOnChange::operator()<Convolution>(ConvolutionStruct& new_value)
{
    LOG_ON_CHANGE_GSH(Convolution);
    if (api::get_compute_pipe_ptr() == nullptr)
        return;

    load_convolution_matrix(new_value);
}

static void load_convolution_matrix(ConvolutionStruct& convo)
{
    if (api::get_convolution().type == UID_CONVOLUTION_TYPE_DEFAULT)
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
    const uint output_width = Holovibes::instance().get_gpu_output_queue()->get_fd().width;
    const uint output_height = Holovibes::instance().get_gpu_output_queue()->get_fd().height;
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

} // namespace holovibes