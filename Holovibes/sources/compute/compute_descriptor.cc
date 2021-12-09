#include "compute_descriptor.hh"
#include "user_interface_descriptor.hh"

#include "holovibes.hh"
#include "tools.hh"
#include "API.hh"

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

float ComputeDescriptor::get_truncate_contrast_max(const int precision) const
{
    float value = GSH::instance().get_contrast_max();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

float ComputeDescriptor::get_truncate_contrast_min(const int precision) const
{
    float value = GSH::instance().get_contrast_min();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

void ComputeDescriptor::check_p_limits()
{
    int upper_bound = GSH::instance().get_time_transformation_size() - 1;

    if (GSH::instance().get_p_accu_level() > upper_bound)
        api::set_p_accu_level(upper_bound);

    upper_bound -= GSH::instance().get_p_accu_level();

    if (upper_bound >= 0 && GSH::instance().get_p_index() > static_cast<uint>(upper_bound))
        api::set_p_index(upper_bound);
}

void ComputeDescriptor::check_q_limits()
{
    int upper_bound = GSH::instance().get_time_transformation_size() - 1;

    if (GSH::instance().get_q_accu_level() > upper_bound)
        api::set_q_accu_level(upper_bound);

    upper_bound -= GSH::instance().get_q_accu_level();

    if (upper_bound >= 0 && GSH::instance().get_q_index() > static_cast<uint>(upper_bound))
        api::set_q_index(upper_bound);
}

void ComputeDescriptor::handle_update_exception()
{
    api::set_p_index(0);
    api::set_time_transformation_size({1});
    api::set_convolution_enabled(false);
}

void ComputeDescriptor::handle_accumulation_exception() { GSH::instance().set_xy_img_accu_level(1); }

void ComputeDescriptor::set_computation_stopped(bool value) { is_computation_stopped = value; }

void ComputeDescriptor::set_weight_rgb(int r, int g, int b)
{
    rgb.weight_r = r;
    rgb.weight_g = g;
    rgb.weight_b = b;
}

void ComputeDescriptor::change_angle()
{
    double rot = GSH::instance().get_rotation();
    double new_rot = (rot == 270.f) ? 0.f : rot + 90.f;

    GSH::instance().set_rotation(new_rot);
}

void ComputeDescriptor::change_flip() { GSH::instance().set_flip_enabled(!GSH::instance().get_flip_enabled()); }

void ComputeDescriptor::reset_windows_display()
{
    lens_view_enabled = false;
    filter2d_view_enabled = false;
    raw_view_enabled = false;
    reticle_display_enabled = false;
}

void ComputeDescriptor::reset_slice_view()
{
    GSH::instance().set_xz_contrast_max(false);
    GSH::instance().set_yz_contrast_max(false);

    GSH::instance().set_xz_log_scale_slice_enabled(false);
    GSH::instance().set_yz_log_scale_slice_enabled(false);

    GSH::instance().set_xz_img_accu_level(1);
    GSH::instance().set_yz_img_accu_level(1);
}

void ComputeDescriptor::set_convolution(bool enable, const std::string& file)
{
    if (enable && file != UID_CONVOLUTION_TYPE_DEFAULT)
        load_convolution_matrix(file);

    convolution_enabled = enable;
}

void ComputeDescriptor::set_divide_by_convo(bool enable)
{
    divide_convolution_enabled = enable && GSH::instance().get_convolution_enabled();
}

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
