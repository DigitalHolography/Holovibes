#pragma once

#include "API.hh"
#include "enum_record_mode.hh"

namespace holovibes::api
{
inline Computation get_compute_mode()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::ComputeMode>().value;
}
inline void set_compute_mode(Computation mode)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ComputeMode{mode});
}

inline float get_pixel_size()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::PixelSize>().value;
}

inline void set_pixel_size(float value)
{
    // std::lock_guard<std::mutex> lock(mutex_);
    holovibes::Holovibes::instance().update_setting(holovibes::settings::PixelSize{value});
}

inline SpaceTransformation get_space_transformation()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::SpaceTransformation>().value;
}

inline TimeTransformation get_time_transformation()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::TimeTransformation>().value;
}

inline ImgType get_img_type()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::ImageType>().value;
}
inline void set_img_type(ImgType type)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ImageType{type});
}

inline uint get_input_buffer_size()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::InputBufferSize>().value;
}
inline void set_input_buffer_size(uint value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::InputBufferSize{value});
}

inline uint get_time_stride() { return holovibes::Holovibes::instance().get_setting<settings::TimeStride>().value; }
inline void set_time_stride(uint value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::TimeStride{value});

    uint batch_size = holovibes::Holovibes::instance().get_setting<settings::BatchSize>().value;

    if (batch_size > value)
        holovibes::Holovibes::instance().update_setting(holovibes::settings::TimeStride{batch_size});
    // Go to lower multiple
    if (value % batch_size != 0)
        holovibes::Holovibes::instance().update_setting(holovibes::settings::TimeStride{value - value % batch_size});
}

inline uint get_batch_size() { return holovibes::Holovibes::instance().get_setting<settings::BatchSize>().value; }
inline bool set_batch_size(uint value)
{
    bool request_time_stride_update = false;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::BatchSize{value});

    if (value > get_input_buffer_size())
        value = get_input_buffer_size();
    uint time_stride = get_time_stride();
    if (time_stride < value)
    {
        holovibes::Holovibes::instance().update_setting(holovibes::settings::TimeStride{value});
        time_stride = value;
        request_time_stride_update = true;
    }
    // Go to lower multiple
    if (time_stride % value != 0)
    {
        set_time_stride(time_stride - time_stride % value);
    }

    return request_time_stride_update;
}

inline uint get_time_transformation_size()
{
    return holovibes::Holovibes::instance().get_setting<settings::TimeTransformationSize>().value;
}

inline void set_time_transformation_size(uint value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::TimeTransformationSize{value});
}

inline float get_lambda() { return holovibes::Holovibes::instance().get_setting<settings::Lambda>().value; }

inline float get_z_distance() { return holovibes::Holovibes::instance().get_setting<settings::ZDistance>().value; }

inline std::vector<float> get_convo_matrix()
{
    return holovibes::Holovibes::instance().get_setting<settings::ConvolutionMatrix>().value;
};

inline void set_convo_matrix(std::vector<float> value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ConvolutionMatrix{value});
}

inline float get_contrast_lower_threshold()
{
    return holovibes::Holovibes::instance().get_setting<settings::ContrastLowerThreshold>().value;
}
inline void set_contrast_lower_threshold(float value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ContrastLowerThreshold{value});
}

inline float get_contrast_upper_threshold()
{
    return holovibes::Holovibes::instance().get_setting<settings::ContrastUpperThreshold>().value;
}
inline void set_contrast_upper_threshold(float value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ContrastUpperThreshold{value});
}

inline uint get_cuts_contrast_p_offset()
{
    return holovibes::Holovibes::instance().get_setting<settings::CutsContrastPOffset>().value;
}
inline void set_cuts_contrast_p_offset(uint value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::CutsContrastPOffset{value});
}

inline unsigned get_renorm_constant()
{
    return holovibes::Holovibes::instance().get_setting<settings::RenormConstant>().value;
}
inline void set_renorm_constant(unsigned int value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RenormConstant{value});
}

inline int get_filter2d_n1() { return holovibes::Holovibes::instance().get_setting<settings::Filter2dN1>().value; }
inline void set_filter2d_n1(int value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2dN1{value});
    set_auto_contrast_all();
}

inline int get_filter2d_n2() { return holovibes::Holovibes::instance().get_setting<settings::Filter2dN2>().value; }
inline void set_filter2d_n2(int value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2dN2{value});
    set_auto_contrast_all();
}

inline int get_filter2d_smooth_low()
{
    return holovibes::Holovibes::instance().get_setting<settings::Filter2dSmoothLow>().value;
}
inline void set_filter2d_smooth_low(int value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2dSmoothLow{value});
}

inline int get_filter2d_smooth_high()
{
    return holovibes::Holovibes::instance().get_setting<settings::Filter2dSmoothHigh>().value;
}
inline void set_filter2d_smooth_high(int value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2dSmoothHigh{value});
}

inline float get_display_rate() { return holovibes::Holovibes::instance().get_setting<settings::DisplayRate>().value; }
inline void set_display_rate(float value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::DisplayRate{value});
}

inline ViewXY get_x(void) { return holovibes::Holovibes::instance().get_setting<settings::X>().value; }

inline uint get_x_cuts() { return holovibes::Holovibes::instance().get_setting<settings::X>().value.start; }

inline int get_x_accu_level()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::X>().value.width;
}

inline ViewXY get_y(void) { return holovibes::Holovibes::instance().get_setting<settings::Y>().value; }

inline uint get_y_cuts() { return holovibes::Holovibes::instance().get_setting<settings::Y>().value.start; }

inline int get_y_accu_level() { return holovibes::Holovibes::instance().get_setting<settings::Y>().value.width; }

// XY
inline ViewXYZ get_xy() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value; }
inline bool get_xy_horizontal_flip()
{
    return holovibes::Holovibes::instance().get_setting<settings::XY>().value.horizontal_flip;
}
inline float get_xy_rotation() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.rotation; }
inline uint get_xy_accumulation_level()
{
    return holovibes::Holovibes::instance().get_setting<settings::XY>().value.output_image_accumulation;
}
inline bool get_xy_log_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::XY>().value.log_enabled;
}
inline bool get_xy_contrast_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::XY>().value.contrast.enabled;
}
inline bool get_xy_contrast_auto_refresh()
{
    return holovibes::Holovibes::instance().get_setting<settings::XY>().value.contrast.auto_refresh;
}
inline bool get_xy_contrast_invert()
{
    return holovibes::Holovibes::instance().get_setting<settings::XY>().value.contrast.invert;
}
inline float get_xy_contrast_min()
{
    return holovibes::Holovibes::instance().get_setting<settings::XY>().value.contrast.min;
}
inline float get_xy_contrast_max()
{
    return holovibes::Holovibes::instance().get_setting<settings::XY>().value.contrast.max;
}
inline bool get_xy_img_accu_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::XY>().value.output_image_accumulation > 1;
}

inline void set_xy(ViewXYZ value) noexcept
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{value});
}

inline void set_xy_horizontal_flip(bool value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.horizontal_flip = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_rotation(float value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.rotation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_accumulation_level(uint value)
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.output_image_accumulation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_log_enabled(bool value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.log_enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_contrast_enabled(bool value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
    pipe_refresh();
}

inline void set_xy_contrast_auto_refresh(bool value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.auto_refresh = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_contrast_invert(bool value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.invert = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_contrast_min(float value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.min = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_contrast_max(float value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.max = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_contrast(float min, float max) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.min = min;
    xy.contrast.max = max;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

// XZ
inline ViewXYZ get_xz() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value; }
inline bool get_xz_horizontal_flip()
{
    return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.horizontal_flip;
}
inline float get_xz_rotation() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.rotation; }
inline uint get_xz_accumulation_level()
{
    return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.output_image_accumulation;
}
inline bool get_xz_log_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.log_enabled;
}
inline bool get_xz_contrast_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.contrast.enabled;
}
inline bool get_xz_contrast_auto_refresh()
{
    return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.contrast.auto_refresh;
}
inline bool get_xz_contrast_invert()
{
    return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.contrast.invert;
}
inline float get_xz_contrast_min()
{
    return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.contrast.min;
}
inline float get_xz_contrast_max()
{
    return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.contrast.max;
}
inline bool get_xz_img_accu_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.output_image_accumulation > 1;
}
inline void set_xz(ViewXYZ value) noexcept
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{value});
}

inline void set_xz_horizontal_flip(bool value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.horizontal_flip = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_rotation(float value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.rotation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_accumulation_level(uint value)
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.output_image_accumulation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_log_enabled(bool value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.log_enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast_enabled(bool value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast_auto_refresh(bool value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.auto_refresh = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast_invert(bool value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.invert = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast_min(float value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.min = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast_max(float value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.max = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast(float min, float max) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.min = min;
    xz.contrast.max = max;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline ViewXYZ get_yz() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value; }
inline bool get_yz_horizontal_flip()
{
    return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.horizontal_flip;
}
inline float get_yz_rotation() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.rotation; }
inline uint get_yz_accumulation_level()
{
    return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.output_image_accumulation;
}
inline bool get_yz_log_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.log_enabled;
}
inline bool get_yz_contrast_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.contrast.enabled;
}
inline bool get_yz_contrast_auto_refresh()
{
    return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.contrast.auto_refresh;
}
inline bool get_yz_contrast_invert()
{
    return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.contrast.invert;
}
inline float get_yz_contrast_min()
{
    return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.contrast.min;
}
inline float get_yz_contrast_max()
{
    return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.contrast.max;
}
inline bool get_yz_img_accu_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.output_image_accumulation > 1;
}
inline void set_yz(ViewXYZ value) noexcept
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{value});
}

inline void set_yz_horizontal_flip(bool value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.horizontal_flip = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_rotation(float value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.rotation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_accumulation_level(uint value)
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.output_image_accumulation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_log_enabled(bool value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.log_enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast_enabled(bool value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast_auto_refresh(bool value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.auto_refresh = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast_invert(bool value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.invert = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast_min(float value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.min = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast_max(float value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.max = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast(float min, float max) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.min = min;
    yz.contrast.max = max;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline ViewWindow get_filter2d() { return holovibes::Holovibes::instance().get_setting<settings::Filter2d>().value; }
inline bool get_filter2d_log_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::Filter2d>().value.log_enabled;
}
inline bool get_filter2d_contrast_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::Filter2d>().value.contrast.enabled;
}
inline bool get_filter2d_contrast_auto_refresh()
{
    return holovibes::Holovibes::instance().get_setting<settings::Filter2d>().value.contrast.auto_refresh;
}
inline bool get_filter2d_contrast_invert()
{
    return holovibes::Holovibes::instance().get_setting<settings::Filter2d>().value.contrast.invert;
}
inline float get_filter2d_contrast_min()
{
    return holovibes::Holovibes::instance().get_setting<settings::Filter2d>().value.contrast.min;
}
inline float get_filter2d_contrast_max()
{
    return holovibes::Holovibes::instance().get_setting<settings::Filter2d>().value.contrast.max;
}

inline void set_filter2d(ViewWindow value) noexcept
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2d{value});
}

inline void set_filter2d_log_enabled(bool value) noexcept
{
    auto filter2d = Holovibes::instance().get_setting<settings::Filter2d>().value;
    filter2d.log_enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2d{filter2d});
}

inline void set_filter2d_contrast_enabled(bool value) noexcept
{
    auto filter2d = Holovibes::instance().get_setting<settings::Filter2d>().value;
    filter2d.contrast.enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2d{filter2d});
}

inline void set_filter2d_contrast_auto_refresh(bool value) noexcept
{
    auto filter2d = Holovibes::instance().get_setting<settings::Filter2d>().value;
    filter2d.contrast.auto_refresh = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2d{filter2d});
}

inline void set_filter2d_contrast_invert(bool value) noexcept
{
    auto filter2d = Holovibes::instance().get_setting<settings::Filter2d>().value;
    filter2d.contrast.invert = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2d{filter2d});
}

inline void set_filter2d_contrast_min(float value) noexcept
{
    auto filter2d = Holovibes::instance().get_setting<settings::Filter2d>().value;
    filter2d.contrast.min = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2d{filter2d});
}

inline void set_filter2d_contrast_max(float value) noexcept
{
    auto filter2d = Holovibes::instance().get_setting<settings::Filter2d>().value;
    filter2d.contrast.max = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2d{filter2d});
}

inline void set_filter2d_contrast(float min, float max) noexcept
{
    auto f2d = Holovibes::instance().get_setting<settings::Filter2d>().value;
    f2d.contrast.min = min;
    f2d.contrast.max = max;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2d{f2d});
}

inline float get_reticle_scale() { return Holovibes::instance().get_setting<settings::ReticleScale>().value; }
inline void set_reticle_scale(float value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ReticleScale{value});
}

inline int get_unwrap_history_size()
{
    return holovibes::Holovibes::instance().get_setting<settings::UnwrapHistorySize>().value;
}

inline bool get_is_computation_stopped()
{
    return holovibes::Holovibes::instance().get_setting<settings::IsComputationStopped>().value;
}
inline void set_is_computation_stopped(bool value)
{
    holovibes::Holovibes::instance().update_setting(settings::IsComputationStopped{value});
}

inline bool get_divide_convolution_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::DivideConvolutionEnabled>().value;
}
inline void set_divide_convolution_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::DivideConvolutionEnabled{value});
}

inline bool get_renorm_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::RenormEnabled>().value;
}
inline void set_renorm_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RenormEnabled{value});
}

inline bool get_fft_shift_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::FftShiftEnabled>().value;
}

inline void set_fft_shift_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::FftShiftEnabled{value});
    pipe_refresh();
}

inline bool get_raw_view_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::RawViewEnabled>().value;
}

inline void set_raw_view_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RawViewEnabled{value});
}

inline bool get_contrast_enabled() { return GSH::instance().get_contrast_enabled(); }

inline bool get_contrast_auto_refresh() { return GSH::instance().get_contrast_auto_refresh(); }

inline bool get_contrast_invert() { return GSH::instance().get_contrast_invert(); }

inline bool get_filter2d_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::Filter2dEnabled>().value;
}
inline void set_filter2d_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::Filter2dEnabled{value});
}

inline bool get_filter2d_view_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::Filter2dViewEnabled>().value;
}

inline bool get_cuts_view_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::CutsViewEnabled>().value;
}
inline void set_cuts_view_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::CutsViewEnabled{value});
}

inline bool get_lens_view_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::LensViewEnabled>().value;
}
inline void set_lens_view_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::LensViewEnabled{value});
}

inline bool get_chart_display_enabled()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::ChartDisplayEnabled>().value;
}

inline bool get_reticle_display_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::ReticleDisplayEnabled>().value;
}
inline void set_reticle_display_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ReticleDisplayEnabled{value});
}

inline uint get_file_buffer_size()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::FileBufferSize>().value;
}

inline void set_file_buffer_size(uint value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::FileBufferSize{value});
}

inline uint get_record_buffer_size()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::RecordBufferSize>().value;
}
inline void set_record_buffer_size(uint value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RecordBufferSize{value});
}

inline uint get_time_transformation_cuts_output_buffer_size()
{
    return holovibes::Holovibes::instance()
        .get_setting<holovibes::settings::TimeTransformationCutsOutputBufferSize>()
        .value;
}
inline void set_time_transformation_cuts_output_buffer_size(uint value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::TimeTransformationCutsOutputBufferSize{value});
}

inline bool get_batch_enabled()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::BatchEnabled>().value;
}

inline void set_batch_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::BatchEnabled{value});
}

inline std::optional<std::string> get_batch_file_path()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::BatchFilePath>().value;
}

inline void set_batch_file_path(std::string value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::BatchFilePath{value});
}

inline std::string get_record_file_path()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::RecordFilePath>().value;
}

inline void set_record_file_path(std::string value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RecordFilePath{value});
}

inline std::optional<size_t> get_record_frame_count()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::RecordFrameCount>().value;
}

inline void set_record_frame_count(std::optional<size_t> value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RecordFrameCount{value});
}

inline RecordMode get_record_mode()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::RecordMode>().value;
}

inline void set_record_mode(RecordMode value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RecordMode{value});
}

inline size_t get_record_frame_skip()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::RecordFrameSkip>().value;
}

inline void set_record_frame_skip(size_t value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RecordFrameSkip{value});
}

inline size_t get_output_buffer_size()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::OutputBufferSize>().value;
}

inline void set_output_buffer_size(size_t value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::OutputBufferSize{value});
}

inline uint get_input_fps()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::InputFPS>().value;
}

inline void set_input_fps(uint value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::InputFPS{value});
}

inline std::string get_input_file_path()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::InputFilePath>().value;
}

inline void set_input_file_path(std::string value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::InputFilePath{value});
}

inline bool get_loop_on_input_file()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::LoopOnInputFile>().value;
}

inline void set_loop_on_input_file(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::LoopOnInputFile{value});
}

inline bool get_load_file_in_gpu()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::LoadFileInGPU>().value;
}

inline void set_load_file_in_gpu(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::LoadFileInGPU{value});
}

inline size_t get_input_file_start_index()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::InputFileStartIndex>().value;
}

inline void set_input_file_start_index(size_t value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::InputFileStartIndex{value});
}

inline size_t get_input_file_end_index()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::InputFileEndIndex>().value;
}

inline void set_input_file_end_index(size_t value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::InputFileEndIndex{value});
}

inline ViewPQ get_p() { return holovibes::Holovibes::instance().get_setting<holovibes::settings::P>().value; }

inline int get_p_accu_level()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::P>().value.width;
}

inline uint get_p_index() { return holovibes::Holovibes::instance().get_setting<holovibes::settings::P>().value.start; }

inline ViewPQ get_q(void) { return holovibes::Holovibes::instance().get_setting<holovibes::settings::Q>().value; }

inline uint get_q_index() { return holovibes::Holovibes::instance().get_setting<holovibes::settings::Q>().value.start; }

inline uint get_q_accu_level()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::Q>().value.width;
}

inline const camera::FrameDescriptor& get_fd() { return Holovibes::instance().get_gpu_input_queue()->get_fd(); };

inline std::shared_ptr<Pipe> get_compute_pipe() { return Holovibes::instance().get_compute_pipe(); };

inline std::shared_ptr<Queue> get_gpu_output_queue() { return Holovibes::instance().get_gpu_output_queue(); };

inline std::shared_ptr<BatchInputQueue> get_gpu_input_queue() { return Holovibes::instance().get_gpu_input_queue(); };

inline units::RectFd get_signal_zone()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::SignalZone>().value;
};
inline units::RectFd get_noise_zone()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::NoiseZone>().value;
};
inline units::RectFd get_composite_zone()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::CompositeZone>().value;
};
inline units::RectFd get_zoomed_zone()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::ZoomedZone>().value;
};
inline units::RectFd get_reticle_zone()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::ReticleZone>().value;
};

inline void set_signal_zone(const units::RectFd& rect)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::SignalZone{rect});
};
inline void set_noise_zone(const units::RectFd& rect)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::NoiseZone{rect});
};
inline void set_composite_zone(const units::RectFd& rect)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::CompositeZone{rect});
};
inline void set_zoomed_zone(const units::RectFd& rect)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ZoomedZone{rect});
};
inline void set_reticle_zone(const units::RectFd& rect)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ReticleZone{rect});
};

inline void set_frame_record_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::FrameRecordEnabled{value});
}

inline bool get_frame_record_enabled()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::FrameRecordEnabled>().value;
}

inline void set_chart_record_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ChartRecordEnabled{value});
}

inline bool get_chart_record_enabled()
{
    return holovibes::Holovibes::instance().get_setting<settings::ChartRecordEnabled>().value;
}

inline void set_convolution_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ConvolutionEnabled{value});
}

inline bool get_convolution_enabled()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::ConvolutionEnabled>().value;
}

inline void set_filter_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::FilterEnabled{value});
};

inline bool get_filter_enabled()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::FilterEnabled>().value;
};

inline bool get_horizontal_flip() { return GSH::instance().get_horizontal_flip(); }

inline double get_rotation() { return GSH::instance().get_rotation(); }

// Ex composite_cache
inline CompositeKind get_composite_kind() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::CompositeKind>().value;
}

inline bool get_composite_auto_weights() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::CompositeAutoWeights>().value;
}

// RGB
inline CompositeRGB get_rgb() noexcept { return holovibes::Holovibes::instance().get_setting<settings::RGB>().value; }
inline uint get_rgb_p_min() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::RGB>().value.frame_index.min;
}
inline uint get_rgb_p_max() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::RGB>().value.frame_index.max;
}
inline float get_weight_r() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::RGB>().value.weight.r;
}
inline float get_weight_g() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::RGB>().value.weight.g;
}
inline float get_weight_b() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::RGB>().value.weight.b;
}

// HSV
inline CompositeHSV get_hsv() noexcept { return holovibes::Holovibes::instance().get_setting<settings::HSV>().value; }
inline uint get_composite_p_min_h() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.h.frame_index.min;
}
inline uint get_composite_p_max_h() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.h.frame_index.max;
}

inline float get_slider_h_threshold_min() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.h.slider_threshold.min;
}
inline float get_slider_h_threshold_max() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.h.slider_threshold.max;
}

inline float get_composite_low_h_threshold() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.h.threshold.min;
}
inline float get_composite_high_h_threshold() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.h.threshold.max;
}
inline uint get_h_blur_kernel_size() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.h.blur.kernel_size;
}
inline uint get_composite_p_min_s() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.s.frame_index.min;
}
inline uint get_composite_p_max_s() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.s.frame_index.max;
}
inline float get_slider_s_threshold_min() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.s.slider_threshold.min;
}
inline float get_slider_s_threshold_max() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.s.slider_threshold.max;
}
inline float get_composite_low_s_threshold() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.s.threshold.min;
}
inline float get_composite_high_s_threshold() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.s.threshold.max;
}
inline uint get_composite_p_min_v() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.v.frame_index.min;
}
inline uint get_composite_p_max_v() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.v.frame_index.max;
}
inline float get_slider_v_threshold_min() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.v.slider_threshold.min;
}
inline float get_slider_v_threshold_max() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.v.slider_threshold.max;
}
inline float get_composite_low_v_threshold() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.v.threshold.min;
}
inline float get_composite_high_v_threshold() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.v.threshold.max;
}
inline bool get_h_blur_activated() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.h.blur.enabled;
}
inline bool get_composite_p_activated_s() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.s.frame_index.activated;
}
inline bool get_composite_p_activated_v() noexcept
{
    return holovibes::Holovibes::instance().get_setting<settings::HSV>().value.v.frame_index.activated;
}
inline uint get_composite_p_red()
{
    return holovibes::Holovibes::instance().get_setting<settings::RGB>().value.frame_index.min;
}
inline uint get_composite_p_blue()
{
    return holovibes::Holovibes::instance().get_setting<settings::RGB>().value.frame_index.max;
}

inline void set_composite_kind(CompositeKind value)
{
    holovibes::Holovibes::instance().update_setting(settings::CompositeKind{value});
}

inline void set_composite_auto_weights(bool value)
{
    holovibes::Holovibes::instance().update_setting(settings::CompositeAutoWeights{value});
    pipe_refresh();
}

// RGB
inline void set_rgb(CompositeRGB value) { holovibes::Holovibes::instance().update_setting(settings::RGB{value}); }

inline void set_rgb_p(int min, int max, bool notify = false)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.frame_index.min = min;
    rgb.frame_index.max = max;
    holovibes::Holovibes::instance().update_setting(settings::RGB{rgb});
    if (notify)
        GSH::instance().set_rgb_p();
}

inline void set_weight_r(double value)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.weight.r = value;
    Holovibes::instance().update_setting(settings::RGB{rgb});
}
inline void set_weight_g(double value)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.weight.g = value;
    Holovibes::instance().update_setting(settings::RGB{rgb});
}
inline void set_weight_b(double value)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.weight.b = value;
    Holovibes::instance().update_setting(settings::RGB{rgb});
}

inline void set_weight_rgb(double r, double g, double b)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.weight.r = r;
    rgb.weight.g = g;
    rgb.weight.b = b;
    Holovibes::instance().update_setting(settings::RGB{rgb});
}

// HSV
inline void set_composite_p_h(int min, int max, bool notify = false)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.frame_index.min = min;
    hsv.h.frame_index.max = max;
    holovibes::Holovibes::instance().update_setting(settings::HSV{hsv});
    if (notify)
        GSH::instance().set_composite_p_h();
}

inline void set_hsv(CompositeHSV value) { holovibes::Holovibes::instance().update_setting(settings::HSV{value}); }
inline void set_slider_h_threshold_min(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.slider_threshold.min = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_slider_h_threshold_max(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.slider_threshold.max = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_low_h_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.threshold.min = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_high_h_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.threshold.max = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_p_min_h(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.frame_index.min = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_p_max_h(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.frame_index.max = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_h_blur_kernel_size(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.blur.kernel_size = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_p_min_s(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.frame_index.min = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_p_max_s(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.frame_index.max = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_slider_s_threshold_min(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.slider_threshold.min = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_slider_s_threshold_max(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.slider_threshold.max = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_low_s_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.threshold.min = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_high_s_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.threshold.max = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_p_min_v(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.frame_index.min = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_p_max_v(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.frame_index.max = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_slider_v_threshold_min(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.slider_threshold.min = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_slider_v_threshold_max(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.slider_threshold.max = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_low_v_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.threshold.min = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_high_v_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.threshold.max = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_h_blur_activated(bool value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.blur.enabled = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_p_activated_s(bool value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.frame_index.activated = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}
inline void set_composite_p_activated_v(bool value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.frame_index.activated = value;
    Holovibes::instance().update_setting(settings::HSV{hsv});
}

inline std::optional<size_t> get_nb_frames_to_record() { return holovibes::Holovibes::instance().get_setting<settings::RecordFrameCount>().value; }
inline void set_nb_frames_to_record(std::optional<size_t> nb_frames)
{
    holovibes::Holovibes::instance().update_setting(settings::RecordFrameCount{nb_frames});
}
#pragma endregion

} // namespace holovibes::api
