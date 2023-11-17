/*! \file
 *
 */

// clang-format off

#define GET_VIEW_MEMBER(type, member)                                        \
    ({                                                                       \
        type result;                                                         \
        auto window = api::get_current_window_type();                        \
        if (window == WindowKind::Filter2D)                                  \
            result = api::get_filter2d().member;                             \
        else                                                                 \
            result = GET_XYZ_MEMBER(type, member);                           \
        result;                                                              \
    })                                                                       \

#define GET_XYZ_MEMBER(type, member)                                         \
    ({                                                                       \
        type result;                                                         \
        auto window = api::get_current_window_type();                        \
        if (window == WindowKind::XYview) {                                  \
            result = api::get_xy().member;                                   \
        } else if (window == WindowKind::XZview) {                           \
            result = api::get_xz().member;                                   \
        } else {                                                             \
            result = api::get_yz().member;                                   \
        }                                                                    \
        result;                                                              \
    })

#define SET_XYZ_MEMBER(member, value)                                        \
    ({                                                                       \
        auto window = api::get_current_window_type();                        \
        if (window == WindowKind::XYview) {                                  \
            api::set_xy_##member(value);                                     \
        } else if (window == WindowKind::XZview) {                           \
            api::set_xz_##member(value);                                     \
        } else {                                                             \
            api::set_yz_##member(value);                                     \
        }                                                                    \
    })
// clang-format on

#include <iostream>
#include "global_state_holder.hh"

#include "holovibes.hh"
#include "API.hh"

namespace holovibes
{
static inline const std::filesystem::path dir(get_exe_dir());

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
    return types.contains(api::get_current_window_type());
}

float GSH::get_contrast_min() const
{
    return GET_VIEW_MEMBER(bool, log_enabled) ? GET_VIEW_MEMBER(float, contrast.min)
                                              : log10(GET_VIEW_MEMBER(float, contrast.min));
}

float GSH::get_contrast_max() const
{
    return GET_VIEW_MEMBER(bool, log_enabled) ? GET_VIEW_MEMBER(float, contrast.max)
                                              : log10(GET_VIEW_MEMBER(float, contrast.max));
}

double GSH::get_rotation() const
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    return GET_XYZ_MEMBER(double, rotation);
}

bool GSH::get_horizontal_flip() const
{

    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    return GET_XYZ_MEMBER(double, horizontal_flip);
}

bool GSH::get_log_enabled() const { return GET_VIEW_MEMBER(bool, log_enabled); }

unsigned GSH::get_accumulation_level() const
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    return GET_XYZ_MEMBER(double, output_image_accumulation);
}

bool GSH::get_contrast_enabled() const noexcept
{
    return GET_VIEW_MEMBER(bool, contrast.enabled);
}

bool GSH::get_contrast_auto_refresh() const noexcept
{
    return GET_VIEW_MEMBER(bool, contrast.auto_refresh);
}

bool GSH::get_contrast_invert() const noexcept {
    return GET_VIEW_MEMBER(bool, contrast.invert);
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

void GSH::set_contrast_enabled(bool value)
{
    auto window = api::get_current_window_type();
    //TODO filter2d
    if (window == WindowKind::Filter2D)
        api::set_filter2d_contrast_enabled(value);
    else
        SET_XYZ_MEMBER(contrast_enabled, value);
}

void GSH::set_contrast_auto_refresh(bool value)
{
    auto window = api::get_current_window_type();
    if (window == WindowKind::Filter2D)
        api::set_filter2d_contrast_auto_refresh(value);
    else
        SET_XYZ_MEMBER(contrast_auto_refresh, value);
}

void GSH::set_contrast_invert(bool value) { 
    auto window = api::get_current_window_type();
    if (window == WindowKind::Filter2D)
        api::set_filter2d_contrast_invert(value);
    else
        SET_XYZ_MEMBER(contrast_invert, value);
}

void GSH::set_contrast_min(float value)
{
    auto window = api::get_current_window_type();
    value = api::get_current_window().log_enabled ? value : pow(10, value);
    if (window == WindowKind::Filter2D)
        api::set_filter2d_contrast_min(value);
    else
        SET_XYZ_MEMBER(contrast_min, value);
}

void GSH::set_contrast_max(float value)
{
    auto window = api::get_current_window_type();
    value = api::get_current_window().log_enabled ? value : pow(10, value);
    if (window == WindowKind::Filter2D)
        api::set_filter2d_contrast_max(value);
    else
        SET_XYZ_MEMBER(contrast_max, value);
}

void GSH::set_log_enabled(bool value)
{
    auto window = api::get_current_window_type();
    if (window == WindowKind::Filter2D)
        api::set_filter2d_log_enabled(value);
    else
        SET_XYZ_MEMBER(log_enabled, value);
}

void GSH::set_accumulation_level(int value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");
    SET_XYZ_MEMBER(accumulation_level, value);
}

void GSH::set_rotation(double value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    SET_XYZ_MEMBER(rotation, value);
}

void GSH::set_horizontal_flip(double value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    SET_XYZ_MEMBER(horizontal_flip, value);
}

void GSH::set_composite_p_h(Span<uint> span, bool notify)
{
    // FIXME - RENAME
    composite_cache_.get_hsv_ref()->h.frame_index.min = span.min;
    composite_cache_.get_hsv_ref()->h.frame_index.max = span.max;
    if (notify)
        this->notify();
}

void GSH::set_rgb_p(Span<int> span, bool notify)
{
    composite_cache_.get_rgb_ref()->frame_index.min = span.min;
    composite_cache_.get_rgb_ref()->frame_index.max = span.max;
    if (notify)
        this->notify();
}

void GSH::set_weight_rgb(double r, double g, double b)
{
    set_weight_r(r);
    set_weight_g(g);
    set_weight_b(b);

    notify();
}

static void load_convolution_matrix(std::shared_ptr<std::vector<float>> convo_matrix, const std::string& file)
{
    auto& holo = Holovibes::instance();

    try
    {
        auto path_file = dir / "convolution_kernels" / file;
        std::string path = path_file.string();

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
        LOG_ERROR("Couldn't load convolution matrix : {}", e.what());
    }
}

void GSH::enable_convolution(std::optional<std::string> file)
{
    compute_cache_.set_convolution_enabled(true);
    compute_cache_.get_convo_matrix_ref()->clear();

    // There is no file None.txt for convolution
    if (file && file.value() != "None")
        load_convolution_matrix(compute_cache_.get_convo_matrix_ref(), file.value());
}

void GSH::set_convolution_enabled(bool value) { compute_cache_.set_convolution_enabled(value); }

void GSH::disable_convolution()
{
    compute_cache_.get_convo_matrix_ref()->clear();
    compute_cache_.set_convolution_enabled(false);
}

#pragma endregion

void GSH::update_contrast(WindowKind kind, float min, float max)
{
    auto window = api::get_current_window_type();

    switch (window)
    {
    case WindowKind::XYview:
        api::set_xy_contrast_min(min);
        api::set_xy_contrast_max(max);
        break;
    case WindowKind::XZview:
        api::set_xz_contrast_min(min);
        api::set_xz_contrast_max(max);
        break;
    case WindowKind::YZview:
        api::set_yz_contrast_min(min);
        api::set_yz_contrast_max(max);
        break;
    // TODO : set_filter2d_contrast_auto
    default:
        api::set_filter2d_contrast_min(min);
        api::set_filter2d_contrast_max(max);
        break;
    }

    notify();
}

/*! \class JsonSettings
 *
 * \brief Struct that help with Json convertion
 *
 */
struct JsonSettings
{

    /*! \brief latest version of holo file version */
    inline static const auto latest_version = GSH::ComputeSettingsVersion::V5;

    /*! \brief path to json patch directories  */
    inline static const auto patches_folder = dir / "json_patches_holofile";

    /*! \brief default convertion function */
    static void convert_default(json& data, const json& json_patch) { data = data.patch(json_patch); }

    /*! \brief convert holo file footer from version 3 to 4 */
    static void convert_v3_to_v4(json& data, const json& json_patch)
    {
        convert_default(data, json_patch);

        data["compute settings"]["image rendering"]["space transformation"] = static_cast<SpaceTransformation>(
            static_cast<int>(data["compute settings"]["image rendering"]["space transformation"]));
        data["compute settings"]["image rendering"]["image mode"] =
            static_cast<Computation>(static_cast<int>(data["compute settings"]["image rendering"]["image mode"]) - 1);
        data["compute settings"]["image rendering"]["time transformation"] = static_cast<TimeTransformation>(
            static_cast<int>(data["compute settings"]["image rendering"]["time transformation"]));
    }

    /*! \brief convert holo file footer from version 4 to 5 */
    static void convert_v4_to_v5(json& data, const json& json_patch)
    {
        if (data.contains("file info"))
        {
            data["info"] = data["file info"];
            data["info"]["input fps"] = 1;
            data["info"]["contiguous"] = 1;
        }

        convert_default(data, json_patch);

        if (data["compute_setting"]["view"]["image_type"] == "PHASEINCREASE")
        {
            data["compute_setting"]["view"]["image_type"] = "PHASE_INCREASE";
        }
        else if (data["compute_setting"]["view"]["image_type"] == "SQUAREDMODULUS")
        {
            data["compute_setting"]["view"]["image_type"] = "SQUARED_MODULUS";
        }
    }

    /*! \class ComputeSettingsConverter
     *
     * \brief Struct that contains all information to perform a convertion
     *
     */
    struct ComputeSettingsConverter
    {
        ComputeSettingsConverter(GSH::ComputeSettingsVersion from,
                                 GSH::ComputeSettingsVersion to,
                                 std::string patch_file,
                                 std::function<void(json&, const json&)> converter = convert_default)
            : from(from)
            , to(to)
            , patch_file(patch_file)
            , converter(converter)
        {
        }

        /*! \brief source version */
        GSH::ComputeSettingsVersion from;

        /*! \brief destination version */
        GSH::ComputeSettingsVersion to;

        /*! \brief patch file name */
        std::string patch_file;

        /*! \brief convertion function */
        std::function<void(json&, const json&)> converter;
    };

    /*! \brief vector that contains all available converters */
    inline static const std::vector<ComputeSettingsConverter> converters = {
        {GSH::ComputeSettingsVersion::V2, GSH::ComputeSettingsVersion::V3, "patch_v2_to_v3.json", convert_default},
        {GSH::ComputeSettingsVersion::V3, GSH::ComputeSettingsVersion::V4, "patch_v3_to_v4.json", convert_v3_to_v4},
        {GSH::ComputeSettingsVersion::V4, GSH::ComputeSettingsVersion::V5, "patch_v4_to_v5.json", convert_v4_to_v5},
    };
};

/*! \brief convert a json based on the source version
 *
 *
 * \param data: json footer
 * \param from: source version
 */
void GSH::convert_json(json& data, GSH::ComputeSettingsVersion from)
{
    auto it = std::find_if(JsonSettings::converters.begin(),
                           JsonSettings::converters.end(),
                           [=](auto converter) -> bool { return converter.from == from; });

    if (it == JsonSettings::converters.end())
        throw std::out_of_range("No converter found");

    std::for_each(it,
                  JsonSettings::converters.end(),
                  [&data](const JsonSettings::ComputeSettingsConverter& converter)
                  {
                      LOG_TRACE("Applying patch version v{}", static_cast<int>(converter.to) + 2);
                      std::ifstream patch_file{JsonSettings::patches_folder / converter.patch_file};
                      try
                      {
                          converter.converter(data, json::parse(patch_file));
                      }
                      catch (const std::exception&)
                      {
                      }
                  });
}

} // namespace holovibes
