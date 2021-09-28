#include "compute_descriptor.hh"

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

ComputeDescriptor& ComputeDescriptor::operator=(const ComputeDescriptor& cd)
{
    is_computation_stopped = is_computation_stopped.load();
    compute_mode = cd.compute_mode.load();
    space_transformation = cd.space_transformation.load();
    time_transformation = cd.time_transformation.load();
    time_transformation_size = cd.time_transformation_size.load();
    pindex = cd.pindex.load();
    p_acc_level = cd.p_acc_level.load();
    p_accu_enabled = cd.p_accu_enabled.load();
    lambda = cd.lambda.load();
    zdistance = cd.zdistance.load();
    img_type = cd.img_type.load();
    unwrap_history_size = cd.unwrap_history_size.load();
    log_scale_slice_xy_enabled = cd.log_scale_slice_xy_enabled.load();
    log_scale_slice_xz_enabled = cd.log_scale_slice_xz_enabled.load();
    log_scale_slice_yz_enabled = cd.log_scale_slice_yz_enabled.load();
    log_scale_filter2d_enabled = cd.log_scale_filter2d_enabled.load();
    fft_shift_enabled = cd.fft_shift_enabled.load();
    contrast_enabled = cd.contrast_enabled.load();
    convolution_enabled = cd.convolution_enabled.load();
    chart_display_enabled = cd.chart_display_enabled.load();
    chart_record_enabled = cd.chart_record_enabled.load();
    contrast_min_slice_xy = cd.contrast_min_slice_xy.load();
    contrast_max_slice_xy = cd.contrast_max_slice_xy.load();
    contrast_min_slice_xz = cd.contrast_min_slice_xz.load();
    contrast_min_slice_yz = cd.contrast_min_slice_yz.load();
    contrast_max_slice_xz = cd.contrast_max_slice_xz.load();
    contrast_max_slice_yz = cd.contrast_max_slice_yz.load();
    contrast_min_filter2d = cd.contrast_min_filter2d.load();
    contrast_max_filter2d = cd.contrast_max_filter2d.load();
    contrast_invert = cd.contrast_invert.load();
    pixel_size = cd.pixel_size.load();
    img_acc_slice_xy_enabled = cd.img_acc_slice_xy_enabled.load();
    img_acc_slice_xz_enabled = cd.img_acc_slice_xz_enabled.load();
    img_acc_slice_yz_enabled = cd.img_acc_slice_yz_enabled.load();
    img_acc_slice_xy_level = cd.img_acc_slice_xy_level.load();
    img_acc_slice_xz_level = cd.img_acc_slice_xz_level.load();
    img_acc_slice_yz_level = cd.img_acc_slice_yz_level.load();
    time_transformation_stride = cd.time_transformation_stride.load();
    time_transformation_cuts_enabled = cd.time_transformation_cuts_enabled.load();
    current_window = cd.current_window.load();
    cuts_contrast_p_offset = cd.cuts_contrast_p_offset.load();
    display_rate = cd.display_rate.load();
    reticle_enabled = cd.reticle_enabled.load();
    reticle_scale = cd.reticle_scale.load();
    signal_zone = cd.signal_zone;
    noise_zone = cd.noise_zone;
    filter2d_enabled = cd.filter2d_enabled.load();
    filter2d_view_enabled = cd.filter2d_view_enabled.load();
    filter2d_n1 = cd.filter2d_n1.load();
    filter2d_n2 = cd.filter2d_n2.load();
    filter2d_smooth_low = cd.filter2d_smooth_low.load();
    filter2d_smooth_high = cd.filter2d_smooth_high.load();
    contrast_auto_refresh = cd.contrast_auto_refresh.load();
    raw_view_enabled = cd.raw_view_enabled.load();
    frame_record_enabled = cd.frame_record_enabled.load();
    return *this;
}

void ComputeDescriptor::signalZone(units::RectFd& rect, AccessMode m)
{
    LockGuard g(mutex_);
    if (m == AccessMode::Get)
    {
        rect = signal_zone;
    }
    else if (m == AccessMode::Set)
    {
        signal_zone = rect;
    }
}

void ComputeDescriptor::noiseZone(units::RectFd& rect, AccessMode m)
{
    LockGuard g(mutex_);
    if (m == AccessMode::Get)
    {
        rect = noise_zone;
    }
    else if (m == AccessMode::Set)
    {
        noise_zone = rect;
    }
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
    switch (current_window)
    {
    case WindowKind::XYview:
        return log_scale_slice_xy_enabled ? contrast_min_slice_xy.load() : log10(contrast_min_slice_xy);
    case WindowKind::XZview:
        return log_scale_slice_xz_enabled ? contrast_min_slice_xz.load() : log10(contrast_min_slice_xz);
    case WindowKind::YZview:
        return log_scale_slice_yz_enabled ? contrast_min_slice_yz.load() : log10(contrast_min_slice_yz);
    case WindowKind::Filter2D:
        return log_scale_filter2d_enabled ? contrast_min_slice_yz.load() : log10(contrast_min_filter2d);
    }
    return 0;
}

float ComputeDescriptor::get_contrast_max() const
{
    switch (current_window)
    {
    case WindowKind::XYview:
        return log_scale_slice_xy_enabled ? contrast_max_slice_xy.load() : log10(contrast_max_slice_xy);
    case WindowKind::XZview:
        return log_scale_slice_xz_enabled ? contrast_max_slice_xz.load() : log10(contrast_max_slice_xz);
    case WindowKind::YZview:
        return log_scale_slice_yz_enabled ? contrast_max_slice_yz.load() : log10(contrast_max_slice_yz);
    case WindowKind::Filter2D:
        return log_scale_filter2d_enabled ? contrast_max_filter2d.load() : log10(contrast_max_filter2d);
    }
    return 0;
}

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

bool ComputeDescriptor::get_img_log_scale_slice_enabled(WindowKind kind) const
{
    switch (kind)
    {
    case WindowKind::XYview:
        return log_scale_slice_xy_enabled;
    case WindowKind::XZview:
        return log_scale_slice_xz_enabled;
    case WindowKind::YZview:
        return log_scale_slice_yz_enabled;
    case WindowKind::Filter2D:
        return log_scale_filter2d_enabled;
    }
    return false;
}

bool ComputeDescriptor::get_img_acc_slice_enabled(WindowKind kind) const
{
    switch (kind)
    {
    case WindowKind::XYview:
        return img_acc_slice_xy_enabled;
    case WindowKind::XZview:
        return img_acc_slice_xz_enabled;
    case WindowKind::YZview:
        return img_acc_slice_yz_enabled;
    }
    return false;
}

unsigned ComputeDescriptor::get_img_acc_slice_level(WindowKind kind) const
{
    switch (kind)
    {
    case WindowKind::XYview:
        return img_acc_slice_xy_level;
    case WindowKind::XZview:
        return img_acc_slice_xz_level;
    case WindowKind::YZview:
        return img_acc_slice_yz_level;
    }
    return 0;
}

void ComputeDescriptor::set_contrast_min(float value)
{
    switch (current_window)
    {
    case WindowKind::XYview:
        contrast_min_slice_xy = log_scale_slice_xy_enabled ? value : pow(10, value);
        break;
    case WindowKind::XZview:
        contrast_min_slice_xz = log_scale_slice_xz_enabled ? value : pow(10, value);
        break;
    case WindowKind::YZview:
        contrast_min_slice_yz = log_scale_slice_yz_enabled ? value : pow(10, value);
        break;
    case WindowKind::Filter2D:
        contrast_min_filter2d = log_scale_filter2d_enabled ? value : pow(10, value);
        break;
    }
}

void ComputeDescriptor::set_contrast_max(float value)
{
    switch (current_window)
    {
    case WindowKind::XYview:
        contrast_max_slice_xy = log_scale_slice_xy_enabled ? value : pow(10, value);
        break;
    case WindowKind::XZview:
        contrast_max_slice_xz = log_scale_slice_xz_enabled ? value : pow(10, value);
        break;
    case WindowKind::YZview:
        contrast_max_slice_yz = log_scale_slice_yz_enabled ? value : pow(10, value);
        break;
    case WindowKind::Filter2D:
        contrast_max_filter2d = log_scale_filter2d_enabled ? value : pow(10, value);
        break;
    }
}

void ComputeDescriptor::set_log_scale_slice_enabled(WindowKind kind, bool value)
{
    switch (kind)
    {
    case WindowKind::XYview:
        log_scale_slice_xy_enabled = value;
        break;
    case WindowKind::XZview:
        log_scale_slice_xz_enabled = value;
        break;
    case WindowKind::YZview:
        log_scale_slice_yz_enabled = value;
        break;
    case WindowKind::Filter2D:
        log_scale_filter2d_enabled = value;
        break;
    }
}

void ComputeDescriptor::set_accumulation(bool value)
{
    switch (current_window)
    {
    case WindowKind::XYview:
        img_acc_slice_xy_enabled = value;
        break;
    case WindowKind::XZview:
        img_acc_slice_xz_enabled = value;
        break;
    case WindowKind::YZview:
        img_acc_slice_yz_enabled = value;
        break;
    }
}

void ComputeDescriptor::set_accumulation_level(float value)
{
    switch (current_window)
    {
    case WindowKind::XYview:
        img_acc_slice_xy_level = value;
        break;
    case WindowKind::XZview:
        img_acc_slice_xz_level = value;
        break;
    case WindowKind::YZview:
        img_acc_slice_yz_level = value;
        break;
    }
}

void ComputeDescriptor::check_p_limits()
{
    uint upper_bound = time_transformation_size - 1;

    if (p_acc_level > upper_bound)
    {
        p_acc_level = upper_bound;
    }

    if (p_accu_enabled)
    {
        upper_bound -= p_acc_level;
    }

    if (pindex > upper_bound)
    {
        pindex = upper_bound;
    }
}

void ComputeDescriptor::check_q_limits()
{
    uint upper_bound = time_transformation_size - 1;

    if (q_acc_level > upper_bound)
    {
        q_acc_level = upper_bound;
    }

    if (q_acc_enabled)
    {
        upper_bound -= q_acc_level;
    }

    if (q_index > upper_bound)
    {
        q_index = upper_bound;
    }
}

void ComputeDescriptor::check_batch_size_limit(const uint input_queue_capacity)
{
    if (batch_size > input_queue_capacity)
    {
        batch_size = input_queue_capacity;
    }
}

void ComputeDescriptor::set_contrast_mode(bool value)
{
    contrast_enabled = value;
    contrast_auto_refresh = true;
}

void ComputeDescriptor::handle_update_exception()
{
    pindex = 0;
    time_transformation_size = 1;
    convolution_enabled = false;
}

void ComputeDescriptor::handle_accumulation_exception()
{
    img_acc_slice_xy_enabled = false;
    img_acc_slice_xy_level = 1;
}

void ComputeDescriptor::set_convolution(bool enable, const std::string& file)
{
    if (enable)
    {
        load_convolution_matrix(file);
        convolution_enabled = true;
    }
    else
    {
        convolution_enabled = false;
        divide_convolution_enabled = false;
    }
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
