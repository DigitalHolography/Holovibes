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

// FIXME
ComputeDescriptor& ComputeDescriptor::operator=(const ComputeDescriptor& cd)
{
    is_computation_stopped = is_computation_stopped.load();
    compute_mode = cd.compute_mode.load();
    space_transformation = cd.space_transformation.load();
    time_transformation = cd.time_transformation.load();
    time_transformation_size = cd.time_transformation_size.load();
    p.index = cd.p.index.load();
    p.accu_level = cd.p.accu_level.load();
    p.accu_enabled = cd.p.accu_enabled.load();
    lambda = cd.lambda.load();
    zdistance = cd.zdistance.load();
    img_type = cd.img_type.load();
    unwrap_history_size = cd.unwrap_history_size.load();
    xy.log_scale_slice_enabled = cd.xy.log_scale_slice_enabled.load();
    xz.log_scale_slice_enabled = cd.xz.log_scale_slice_enabled.load();
    yz.log_scale_slice_enabled = cd.yz.log_scale_slice_enabled.load();
    fft_shift_enabled = cd.fft_shift_enabled.load();
    xy.contrast_enabled = cd.xy.contrast_enabled.load();
    convolution_enabled = cd.convolution_enabled.load();
    chart_display_enabled = cd.chart_display_enabled.load();
    chart_record_enabled = cd.chart_record_enabled.load();
    xy.contrast_min_slice = cd.xy.contrast_min_slice.load();
    xy.contrast_max_slice = cd.xy.contrast_max_slice.load();
    xz.contrast_min_slice = cd.xz.contrast_min_slice.load();
    yz.contrast_min_slice = cd.yz.contrast_min_slice.load();
    xz.contrast_max_slice = cd.xz.contrast_max_slice.load();
    yz.contrast_max_slice = cd.yz.contrast_max_slice.load();
    filter2d.contrast_min_slice = cd.filter2d.contrast_min_slice.load();
    filter2d.contrast_max_slice = cd.filter2d.contrast_max_slice.load();
    xy.contrast_invert = cd.xy.contrast_invert.load();
    pixel_size = cd.pixel_size.load();
    xy.img_acc_slice_enabled = cd.xy.img_acc_slice_enabled.load();
    xz.img_acc_slice_enabled = cd.xz.img_acc_slice_enabled.load();
    yz.img_acc_slice_enabled = cd.yz.img_acc_slice_enabled.load();
    xy.img_acc_slice_level = cd.xy.img_acc_slice_level.load();
    xz.img_acc_slice_level = cd.xz.img_acc_slice_level.load();
    yz.img_acc_slice_level = cd.yz.img_acc_slice_level.load();
    time_transformation_stride = cd.time_transformation_stride.load();
    time_transformation_cuts_enabled = cd.time_transformation_cuts_enabled.load();
    current_window = cd.current_window.load();
    cuts_contrast_p_offset = cd.cuts_contrast_p_offset.load();
    display_rate = cd.display_rate.load();
    reticle_view_enabled = cd.reticle_view_enabled.load();
    reticle_scale = cd.reticle_scale.load();
    signal_zone = cd.signal_zone;
    noise_zone = cd.noise_zone;
    filter2d_enabled = cd.filter2d_enabled.load();
    filter2d_view_enabled = cd.filter2d_view_enabled.load();
    filter2d_n1 = cd.filter2d_n1.load();
    filter2d_n2 = cd.filter2d_n2.load();
    filter2d_smooth_low = cd.filter2d_smooth_low.load();
    filter2d_smooth_high = cd.filter2d_smooth_high.load();
    xy.contrast_auto_refresh = cd.xy.contrast_auto_refresh.load();
    raw_view_enabled = cd.raw_view_enabled.load();
    frame_record_enabled = cd.frame_record_enabled.load();
    return *this;
}

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

bool ComputeDescriptor::get_contrast_enabled() const { return current->contrast_enabled; }
bool ComputeDescriptor::get_contrast_auto_refresh() const { return current->contrast_auto_refresh; }
bool ComputeDescriptor::get_contrast_invert_enabled() const { return current->contrast_invert; }
float ComputeDescriptor::get_contrast_min() const
{
    return current->log_scale_slice_enabled ? current->contrast_min_slice.load() : log10(current->contrast_min_slice);
}

float ComputeDescriptor::get_contrast_max() const
{
    return current->log_scale_slice_enabled ? current->contrast_max_slice.load() : log10(current->contrast_max_slice);
}

bool ComputeDescriptor::get_img_log_scale_slice_enabled() const { return current->log_scale_slice_enabled; }

bool ComputeDescriptor::get_img_acc_slice_enabled() const
{
    return reinterpret_cast<XY_XZ_YZ_WindowView*>(current)->img_acc_slice_enabled;
}

unsigned ComputeDescriptor::get_img_acc_slice_level() const
{
    return reinterpret_cast<XY_XZ_YZ_WindowView*>(current)->img_acc_slice_level;
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

void ComputeDescriptor::set_contrast_mode(bool value) { current->contrast_enabled = value; }
void ComputeDescriptor::set_contrast_invert(bool value) { current->contrast_invert = value; }
void ComputeDescriptor::set_contrast_auto_refresh(bool value) { current->contrast_auto_refresh = value; }
void ComputeDescriptor::set_contrast_enabled(bool value) { current->contrast_enabled = value; }

void ComputeDescriptor::set_contrast_min(float value)
{
    current->contrast_min_slice = current->log_scale_slice_enabled ? value : pow(10, value);
}

void ComputeDescriptor::set_contrast_max(float value)
{
    current->contrast_max_slice = current->log_scale_slice_enabled ? value : pow(10, value);
}

void ComputeDescriptor::set_log_scale_slice_enabled(bool value) { current->log_scale_slice_enabled = value; }

void ComputeDescriptor::set_accumulation(bool value)
{
    reinterpret_cast<XY_XZ_YZ_WindowView*>(current)->img_acc_slice_enabled = value;
}

void ComputeDescriptor::set_accumulation_level(float value)
{
    reinterpret_cast<XY_XZ_YZ_WindowView*>(current)->img_acc_slice_level = value;
}

void ComputeDescriptor::check_p_limits()
{
    uint upper_bound = time_transformation_size - 1;

    if (p.accu_level > upper_bound)
        p.accu_level = upper_bound;

    if (p.accu_enabled)
        upper_bound -= p.accu_level;

    if (p.index > upper_bound)
        p.index = upper_bound;
}

void ComputeDescriptor::check_q_limits()
{
    uint upper_bound = time_transformation_size - 1;

    if (q.accu_level > upper_bound)
        q.accu_level = upper_bound;

    if (q.accu_enabled)
        upper_bound -= q.accu_level;

    if (q.index > upper_bound)
        q.index = upper_bound;
}

void ComputeDescriptor::check_batch_size_limit(const uint input_queue_capacity)
{
    if (batch_size > input_queue_capacity)
        batch_size = input_queue_capacity;
}

void ComputeDescriptor::set_compute_mode(Computation mode) { compute_mode = mode; }

void ComputeDescriptor::set_space_transformation_from_string(const std::string& value)
{
    if (value == "None")
        space_transformation = SpaceTransformation::None;
    else if (value == "1FFT")
        space_transformation = SpaceTransformation::FFT1;
    else if (value == "2FFT")
        space_transformation = SpaceTransformation::FFT2;
    else
    {
        // Shouldn't happen
        space_transformation = SpaceTransformation::None;
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

void ComputeDescriptor::handle_accumulation_exception()
{
    xy.img_acc_slice_enabled = false;
    xy.img_acc_slice_level = 1;
}

void ComputeDescriptor::set_time_transformation_stride(int value) { time_transformation_stride = value; }

void ComputeDescriptor::set_time_transformation_size(int value) { time_transformation_size = value; }

void ComputeDescriptor::set_batch_size(int value) { batch_size = value; }

void ComputeDescriptor::set_convolution_enabled(bool value) { convolution_enabled = value; }

void ComputeDescriptor::set_divide_convolution_mode(bool value) { divide_convolution_enabled = value; }

void ComputeDescriptor::set_reticle_view_enabled(bool value) { reticle_view_enabled = value; }

void ComputeDescriptor::set_reticle_scale(double value) { reticle_scale = value; }

void ComputeDescriptor::set_img_type(ImgType type) { img_type = type; }

void ComputeDescriptor::set_computation_stopped(bool value) { is_computation_stopped = value; }

void ComputeDescriptor::set_time_transformation_cuts_enabled(bool value) { time_transformation_cuts_enabled = value; }

void ComputeDescriptor::set_renorm_enabled(bool value) { renorm_enabled = value; }

void ComputeDescriptor::set_filter2d_enabled(bool value) { filter2d_enabled = value; }

void ComputeDescriptor::set_filter2d_n1(int n) { filter2d_n1 = n; }

void ComputeDescriptor::set_filter2d_n2(int n) { filter2d_n2 = n; }

void ComputeDescriptor::set_fft_shift_enabled(bool value) { fft_shift_enabled = value; }

void ComputeDescriptor::set_lens_view_enabled(bool value) { lens_view_enabled = value; }

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

void ComputeDescriptor::set_p_index(int value) { p.index = value; }

void ComputeDescriptor::set_q_index(int value) { q.index = value; }

void ComputeDescriptor::set_lambda(float value) { lambda = value; }

void ComputeDescriptor::set_zdistance(float value) { zdistance = value; }

void ComputeDescriptor::set_rgb_p_min(int value) { rgb_p_min = value; }

void ComputeDescriptor::set_rgb_p_max(int value) { rgb_p_max = value; }

void ComputeDescriptor::set_composite_p_min_h(int value) { composite_p_min_h = value; }

void ComputeDescriptor::set_composite_p_max_h(int value) { composite_p_max_h = value; }

void ComputeDescriptor::set_composite_p_min_s(int value) { composite_p_min_s = value; }

void ComputeDescriptor::set_composite_p_max_s(int value) { composite_p_max_s = value; }

void ComputeDescriptor::set_composite_p_min_v(int value) { composite_p_min_v = value; }

void ComputeDescriptor::set_composite_p_max_v(int value) { composite_p_max_v = value; }

void ComputeDescriptor::set_weight_rgb(int r, int g, int b)
{
    weight_r = r;
    weight_g = g;
    weight_b = b;
}

void ComputeDescriptor::set_composite_auto_weights(bool value) { composite_auto_weights = value; }

void ComputeDescriptor::set_composite_kind(CompositeKind kind) { composite_kind = kind; }

void ComputeDescriptor::set_composite_p_activated_s(bool value) { composite_p_activated_s = value; }

void ComputeDescriptor::set_composite_p_activated_v(bool value) { composite_p_activated_v = value; }

void ComputeDescriptor::set_h_blur_activated(bool value) { h_blur_activated = value; }

void ComputeDescriptor::set_h_blur_kernel_size(int value) { h_blur_kernel_size = value; }

void ComputeDescriptor::set_x_accu(bool enabled, int level)
{
    x.accu_enabled = enabled;
    x.accu_level = level;
}

void ComputeDescriptor::set_y_accu(bool enabled, int level)
{
    y.accu_enabled = enabled;
    y.accu_level = level;
}

void ComputeDescriptor::set_p_accu(bool enabled, int level)
{
    p.accu_enabled = enabled;
    p.accu_level = level;
}

void ComputeDescriptor::set_q_accu(bool enabled, int level)
{
    q.accu_enabled = enabled;
    q.accu_level = level;
}

void ComputeDescriptor::change_angle()
{
    auto w = reinterpret_cast<XY_XZ_YZ_WindowView*>(current);
    w->rot = (w->rot == 270.f) ? 0.f : w->rot + 90.f;
}
void ComputeDescriptor::change_flip()
{
    auto w = reinterpret_cast<XY_XZ_YZ_WindowView*>(current);
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
    reticle_view_enabled = false;
}

void ComputeDescriptor::reset_gui()
{
    p.index = 0;
    time_transformation_size = 1;
}

void ComputeDescriptor::reset_slice_view()
{
    xz.contrast_max_slice = false;
    yz.contrast_max_slice = false;
    xz.log_scale_slice_enabled = false;
    yz.log_scale_slice_enabled = false;
    xz.img_acc_slice_enabled = false;
    yz.img_acc_slice_enabled = false;
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
