#include "compute_descriptor.hh"
#include "user_interface_descriptor.hh"

#include "holovibes.hh"
#include "tools.hh"

namespace holovibes
{
using LockGuard = std::lock_guard<std::mutex>;

ComputeDescriptor::ComputeDescriptor()
    : Observable()
{
}

ComputeDescriptor::~ComputeDescriptor() {}

void ComputeDescriptor::signalZone(units::RectFd& rect, AccessMode m)
{
    LockGuard g(mutex_);
    if (m == AccessMode::Get)
        rect = signal_zone;
    else if (m == AccessMode::Set)
        signal_zone = rect;
}

void ComputeDescriptor::noiseZone(units::RectFd& rect, AccessMode m)
{
    LockGuard g(mutex_);
    if (m == AccessMode::Get)
        rect = noise_zone;
    else if (m == AccessMode::Set)
        noise_zone = rect;
}

units::RectFd ComputeDescriptor::getCompositeZone() const
{
    LockGuard g(mutex_);
    return composite_zone;
}

void ComputeDescriptor::setCompositeZone(const units::RectFd& rect)
{
    LockGuard g(mutex_);
    composite_zone = rect;
}

units::RectFd ComputeDescriptor::getZoomedZone() const
{
    LockGuard g(mutex_);
    return zoomed_zone;
}

void ComputeDescriptor::setZoomedZone(const units::RectFd& rect)
{
    LockGuard g(mutex_);
    zoomed_zone = rect;
}

void ComputeDescriptor::setReticleZone(const units::RectFd& rect)
{
    LockGuard g(mutex_);
    reticle_zone = rect;
}

units::RectFd ComputeDescriptor::getReticleZone() const
{
    LockGuard g(mutex_);
    return reticle_zone;
}

float ComputeDescriptor::get_contrast_min() const
{
    return current->log_scale_slice_enabled ? current->contrast_min.load() : log10(current->contrast_min);
}

float ComputeDescriptor::get_contrast_max() const
{
    return current->log_scale_slice_enabled ? current->contrast_max.load() : log10(current->contrast_max);
}

bool ComputeDescriptor::get_img_log_scale_slice_enabled() const { return current->log_scale_slice_enabled; }

unsigned ComputeDescriptor::get_img_accu_level() const { return reinterpret_cast<View_XYZ*>(current)->img_accu_level; }

float ComputeDescriptor::get_truncate_contrast_max(const int precision) const
{
    float value = get_contrast_max();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

float ComputeDescriptor::get_truncate_contrast_min(const int precision) const
{
    float value = get_contrast_min();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

void ComputeDescriptor::set_contrast_min(float value)
{
    current->contrast_min = current->log_scale_slice_enabled ? value : pow(10, value);
}

void ComputeDescriptor::set_contrast_max(float value)
{
    current->contrast_max = current->log_scale_slice_enabled ? value : pow(10, value);
}

void ComputeDescriptor::set_log_scale_slice_enabled(bool value) { current->log_scale_slice_enabled = value; }

void ComputeDescriptor::set_accumulation_level(int value)
{
    reinterpret_cast<View_XYZ*>(current)->img_accu_level = value;
}

void ComputeDescriptor::check_p_limits()
{
    int upper_bound = time_transformation_size - 1;

    if (p.accu_level > upper_bound)
        p.accu_level = upper_bound;

    upper_bound -= p.accu_level;

    if (upper_bound >= 0 && p.index > static_cast<uint>(upper_bound))
        p.index = upper_bound;
}

void ComputeDescriptor::check_q_limits()
{
    int upper_bound = time_transformation_size - 1;

    if (q.accu_level > upper_bound)
        q.accu_level = upper_bound;

    upper_bound -= q.accu_level;

    if (upper_bound >= 0 && q.index > static_cast<uint>(upper_bound))
        q.index = upper_bound;
}

void ComputeDescriptor::check_batch_size_limit()
{
    if (batch_size > input_buffer_size)
        batch_size = input_buffer_size.load();
}

void ComputeDescriptor::set_space_transformation_from_string(const std::string& value)
{
    if (value == "None")
        space_transformation = SpaceTransformation::NONE;
    else if (value == "1FFT")
        space_transformation = SpaceTransformation::FFT1;
    else if (value == "2FFT")
        space_transformation = SpaceTransformation::FFT2;
    else
    {
        // Shouldn't happen
        space_transformation = SpaceTransformation::NONE;
        LOG_ERROR << "Unknown space transform: " << value << ", falling back to None";
    }
}

void ComputeDescriptor::set_time_transformation_from_string(const std::string& value)
{
    if (value == "STFT")
        time_transformation = TimeTransformation::STFT;
    else if (value == "PCA")
        time_transformation = TimeTransformation::PCA;
    else if (value == "None")
        time_transformation = TimeTransformation::NONE;
    else if (value == "SSA_STFT")
        time_transformation = TimeTransformation::SSA_STFT;
}

void ComputeDescriptor::adapt_time_transformation_stride()
{
    if (time_transformation_stride < batch_size)
        time_transformation_stride = batch_size.load();
    else if (time_transformation_stride % batch_size != 0) // Go to lower multiple
        time_transformation_stride -= time_transformation_stride % batch_size;
}

void ComputeDescriptor::handle_update_exception()
{
    p.index = 0;
    time_transformation_size = 1;
    convolution_enabled = false;
}

void ComputeDescriptor::handle_accumulation_exception() { xy.img_accu_level = 1; }

void ComputeDescriptor::set_computation_stopped(bool value) { is_computation_stopped = value; }

void ComputeDescriptor::set_x_cuts(int value)
{
    auto& holo = Holovibes::instance();
    const auto& fd = holo.get_gpu_input_queue()->get_fd();
    if (value < fd.width)
        x.cuts = value;
}

void ComputeDescriptor::set_y_cuts(int value)
{
    auto& holo = Holovibes::instance();
    const auto& fd = holo.get_gpu_input_queue()->get_fd();
    if (value < fd.height)
        y.cuts = value;
}

void ComputeDescriptor::set_weight_rgb(int r, int g, int b)
{
    rgb.weight_r = r;
    rgb.weight_g = g;
    rgb.weight_b = b;
}

void ComputeDescriptor::change_angle()
{
    auto w = reinterpret_cast<View_XYZ*>(current);
    if (w == nullptr)
    {
        LOG_ERROR << "Current window cannot be rotated.";
        return;
    }

    w->rot = (w->rot == 270.f) ? 0.f : w->rot + 90.f;
}

void ComputeDescriptor::change_flip()
{
    auto w = reinterpret_cast<View_XYZ*>(current);
    if (w == nullptr)
    {
        LOG_ERROR << "Current window cannot be flipped.";
        return;
    }

    w->flip_enabled = !w->flip_enabled;
}

void ComputeDescriptor::change_window(int index)
{
    if (index == 0)
    {
        current = &xy;
        current_window = WindowKind::XYview;
    }
    else if (index == 1)
    {
        current = &xz;
        current_window = WindowKind::XZview;
    }
    else if (index == 2)
    {
        current = &yz;
        current_window = WindowKind::YZview;
    }
    else if (index == 3)
    {
        current = &filter2d;
        current_window = WindowKind::Filter2D;
    }
}

void ComputeDescriptor::set_rendering_params(float value)
{
    time_transformation_stride = std::ceil(value / 20.0f);
    batch_size = 1;
}

void ComputeDescriptor::reset_windows_display()
{
    lens_view_enabled = false;
    filter2d_view_enabled = false;
    raw_view_enabled = false;
    reticle_display_enabled = false;
}

void ComputeDescriptor::reset_slice_view()
{
    xz.contrast_max = false;
    yz.contrast_max = false;
    xz.log_scale_slice_enabled = false;
    yz.log_scale_slice_enabled = false;
    xz.img_accu_level = 1;
    yz.img_accu_level = 1;
}

void ComputeDescriptor::set_convolution(bool enable, const std::string& file)
{
    if (enable && file != UID_CONVOLUTION_TYPE_DEFAULT)
        load_convolution_matrix(file);

    convolution_enabled = enable;
}

void ComputeDescriptor::set_divide_by_convo(bool enable) { divide_convolution_enabled = enable && convolution_enabled; }

void ComputeDescriptor::load_convolution_matrix(const std::string& file)
{
    auto& holo = Holovibes::instance();
    convo_matrix.clear();

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

        convo_matrix.resize(size, 0.0f);

        uint kernel_indice = 0;
        for (uint i = first_row; i < last_row; i++)
        {
            for (uint j = first_col; j < last_col; j++)
            {
                convo_matrix[i * output_width + j] = matrix[kernel_indice];
                kernel_indice++;
            }
        }
    }
    catch (std::exception& e)
    {
        convo_matrix.clear();
        LOG_ERROR << "Couldn't load convolution matrix " << e.what();
    }
}

} // namespace holovibes
