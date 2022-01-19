#include "global_state_holder.hh"

#include "holovibes.hh"
#include "API.hh"
#include "view_struct.hh"
#include "rendering_struct.hh"
#include "composite_struct.hh"
#include "internals_struct.hh"
#include "advanced_struct.hh"

namespace holovibes
{

GSH& GSH::instance()
{
    static GSH* instance_ = nullptr;
    if (instance_ == nullptr)
        instance_ = new GSH();

    return *instance_;
}

#pragma region(collapsed) GETTERS

bool GSH::is_current_window_xyz_type() const
{
    static const std::set<WindowKind> types = {WindowKind::XYview, WindowKind::XZview, WindowKind::YZview};
    return types.contains(view_cache_.get_current_window());
}

float GSH::get_contrast_min() const
{
    return get_current_window().log_enabled ? get_current_window().contrast.min
                                            : log10(get_current_window().contrast.min);
}

float GSH::get_contrast_max() const
{
    return get_current_window().log_enabled ? get_current_window().contrast.max
                                            : log10(get_current_window().contrast.max);
}

double GSH::get_rotation() const
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<const ViewXYZ&>(get_current_window());
    return w.rot;
}

bool GSH::get_flip_enabled() const
{

    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<const ViewXYZ&>(get_current_window());
    return w.flip_enabled;
}

bool GSH::get_img_log_scale_slice_enabled() const { return get_current_window().log_enabled; }

unsigned GSH::get_img_accu_level() const
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<const ViewXYZ&>(get_current_window());
    return w.img_accu_level;
}
#pragma endregion

#pragma region(collapsed) SETTERS

void GSH::set_batch_size(uint value)
{
    if (value > advanced_cache_.get_input_buffer_size())
        value = advanced_cache_.get_input_buffer_size();

    if (compute_cache_.get_time_stride() < value)
        compute_cache_.set_time_stride(value);
    // Go to lower multiple
    if (compute_cache_.get_time_stride() % value != 0)
        compute_cache_.set_time_stride(compute_cache_.get_time_stride() - compute_cache_.get_time_stride() % value);

    compute_cache_.set_batch_size(value);
}

void GSH::set_time_transformation_size(uint value)
{
    // FIXME: temporary fix due to ttsize change in pipe.make_request
    // std::lock_guard<std::mutex> lock(mutex_);
    compute_cache_.set_time_transformation_size(value);
}

void GSH::set_time_stride(uint value)
{
    // FIXME: temporary fix due to ttstride change in pipe.make_request
    // std::lock_guard<std::mutex> lock(mutex_);
    compute_cache_.set_time_stride(value);

    if (compute_cache_.get_batch_size() > value)
        compute_cache_.set_time_stride(compute_cache_.get_batch_size());
    // Go to lower multiple
    if (value % compute_cache_.get_batch_size() != 0)
        compute_cache_.set_time_stride(value - value % compute_cache_.get_batch_size());
}

void GSH::set_contrast_enabled(bool contrast_enabled) { get_current_window()->contrast.enabled = contrast_enabled; }

void GSH::set_contrast_auto_refresh(bool contrast_auto_refresh)
{
    get_current_window()->contrast.auto_refresh = contrast_auto_refresh;
}

void GSH::set_contrast_invert(bool contrast_invert) { get_current_window()->contrast.invert = contrast_invert; }

void GSH::set_contrast_min(float value)
{
    get_current_window()->contrast.min = get_current_window()->log_enabled ? value : pow(10, value);
}

void GSH::set_contrast_max(float value)
{
    get_current_window()->contrast.max = get_current_window()->log_enabled ? value : pow(10, value);
}

void GSH::set_log_scale_slice_enabled(bool value) { get_current_window()->log_enabled = value; }

void GSH::set_accumulation_level(int value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    reinterpret_cast<ViewXYZ*>(get_current_window().get())->img_accu_level = value;
}

void GSH::set_rotation(double value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    reinterpret_cast<ViewXYZ*>(get_current_window().get())->rot = value;
}

void GSH::set_flip_enabled(double value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    reinterpret_cast<ViewXYZ*>(get_current_window().get())->flip_enabled = value;
}

void GSH::set_fft_shift_enabled(bool value)
{
    view_cache_.set_fft_shift_enabled(value);
    api::pipe_refresh();
}

void GSH::set_composite_p_h(Span<uint> span, bool notify)
{
    composite_cache_.get_hsv_ref()->h.p.max = span.min;
    composite_cache_.get_hsv_ref()->h.p.max = span.max;
    if (notify)
        this->notify();
}

void GSH::set_rgb_p(Span<int> span, bool notify)
{
    composite_cache_.get_rgb_ref()->p.min = span.min;
    composite_cache_.get_rgb_ref()->p.max = span.max;
    if (notify)
        this->notify();
}

void GSH::set_weight_rgb(int r, int g, int b)
{
    set_weight_r(r);
    set_weight_g(g);
    set_weight_b(b);
}

static void load_convolution_matrix(std::shared_ptr<std::vector<float>> convo_matrix, const std::string& file)
{
    auto& holo = Holovibes::instance();

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

        convo_matrix->resize(size, 0.0f);

        uint kernel_indice = 0;
        for (uint i = first_row; i < last_row; i++)
        {
            for (uint j = first_col; j < last_col; j++)
            {
                (*convo_matrix)[i * output_width + j] = matrix[kernel_indice];
                kernel_indice++;
            }
        }
    }
    catch (std::exception& e)
    {
        convo_matrix->clear();
        LOG_ERROR(main, "Couldn't load convolution matrix : {}", e.what());
    }
}

void GSH::enable_convolution(std::optional<std::string> file)
{
    compute_cache_.set_convolution_enabled(true);
    compute_cache_.get_convo_matrix_ref()->clear();

    if (file)
        load_convolution_matrix(compute_cache_.get_convo_matrix_ref(), file.value());
}

void GSH::set_convolution_enabled(bool value) { compute_cache_.set_convolution_enabled(value); }

void GSH::disable_convolution()
{
    compute_cache_.get_convo_matrix_ref()->clear();
    compute_cache_.set_convolution_enabled(false);
}

#pragma endregion

/*! \brief Change the window according to the given index */
void GSH::change_window(uint index) { view_cache_.set_current_window(static_cast<WindowKind>(index)); }

void GSH::update_contrast(WindowKind kind, float min, float max)
{
    std::shared_ptr<ViewWindow> window = get_window(kind);
    window->contrast.min = min;
    window->contrast.max = max;

    notify();
}

std::shared_ptr<ViewWindow> GSH::get_window(WindowKind kind)
{
    const std::map<WindowKind, std::shared_ptr<ViewWindow>> kind_window = {
        {WindowKind::XYview, view_cache_.get_xy_ref()},
        {WindowKind::XZview, view_cache_.get_xz_ref()},
        {WindowKind::YZview, view_cache_.get_yz_ref()},
        {WindowKind::Filter2D, view_cache_.get_filter2d_ref()},
    };

    return kind_window.at(kind);
}

const ViewWindow& GSH::get_window(WindowKind kind) const
{
    const std::map<WindowKind, const ViewWindow*> kind_window = {
        {WindowKind::XYview, &view_cache_.get_xy_const_ref()},
        {WindowKind::XZview, &view_cache_.get_xz_const_ref()},
        {WindowKind::YZview, &view_cache_.get_yz_const_ref()},
        {WindowKind::Filter2D, &view_cache_.get_filter2d_const_ref()},
    };

    return *kind_window.at(kind);
}

const ViewWindow& GSH::get_current_window() const { return get_window(view_cache_.get_current_window()); }

/* private */
std::shared_ptr<ViewWindow> GSH::get_current_window() { return get_window(view_cache_.get_current_window()); }

void debug_compute_settings()
{
    std::cout << std::setw(1) << json{ComputeSettings{}};
    std::cout << std::setw(1) << api::compute_settings_to_json();
}

void GSH::load_compute_settings(const std::string& settings_path,
                                ComputeSettingsVersion version = SerializerSettings::latest_version,
                                bool no_converter = false)
{
    auto it = std::find_if(converters.begin(),
                           converters.end(),
                           [=](auto converter) -> bool {
                               return converter.from == version && converter.to == SerializerSettings::internal_version
                           });

    if (!no_converter && it == converters.end())
        throw std::out_of_range("No converter found");

    json patch;
    json settings;

    {
        std::ifstream settings_file{settings_path};
        settings_file >> settings;
    }

    {
        std::ifstream patch_file{SerializerSettings::patches_folder / it->patch_file};
        patch_file >> patch;
    }

    auto cd = settings.patch(patch).get<SerializerSettings::ComputeSettings>();
    // TODO: put settings into the GSH;
}

} // namespace holovibes
