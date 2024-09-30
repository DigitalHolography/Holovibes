#pragma once

#include "API.hh"
#include "enum_record_mode.hh"

namespace holovibes::api
{

static std::vector<std::string> authors{"Titouan Gragnic",
                                        "Arthur Courselle",
                                        "Gustave Herve",
                                        "Alexis Pinson",
                                        "Etienne Senigout",
                                        "Bastien Gaulier",
                                        "Simon Riou",

                                        "Chloé Magnier",
                                        "Noé Topeza",
                                        "Maxime Boy-Arnould",

                                        "Oscar Morand",
                                        "Paul Duhot",
                                        "Thomas Xu",
                                        "Jules Guillou",
                                        "Samuel Goncalves",
                                        "Edgar Delaporte",

                                        "Adrien Langou",
                                        "Julien Nicolle",
                                        "Sacha Bellier",
                                        "David Chemaly",
                                        "Damien Didier",

                                        "Philippe Bernet",
                                        "Eliott Bouhana",
                                        "Fabien Colmagro",
                                        "Marius Dubosc",
                                        "Guillaume Poisson",

                                        "Anthony Strazzella",
                                        "Ilan Guenet",
                                        "Nicolas Blin",
                                        "Quentin Kaci",
                                        "Theo Lepage",

                                        "Loïc Bellonnet-Mottet",
                                        "Antoine Martin",
                                        "François Te",

                                        "Ellena Davoine",
                                        "Clement Fang",
                                        "Danae Marmai",
                                        "Hugo Verjus",

                                        "Eloi Charpentier",
                                        "Julien Gautier",
                                        "Florian Lapeyre",

                                        "Thomas Jarrossay",
                                        "Alexandre Bartz",

                                        "Cyril Cetre",
                                        "Clement Ledant",

                                        "Eric Delanghe",
                                        "Arnaud Gaillard",
                                        "Geoffrey Le Gourrierec",

                                        "Jeffrey Bencteux",
                                        "Thomas Kostas",
                                        "Pierre Pagnoux",

                                        "Antoine Dillée",
                                        "Romain Cancillière",

                                        "Michael Atlan"};

constexpr std::vector<std::string> get_credits()
{
    std::vector<std::string> res{"", "", ""};

    size_t nb_columns = 3;
    for (size_t i = 0; i < authors.size(); i++)
        res[i % nb_columns] += authors[i] + "<br>";

    return res;
}

inline Computation get_compute_mode() { return GET_SETTING(ComputeMode); }

inline void set_compute_mode(Computation mode) { UPDATE_SETTING(ComputeMode, mode); }

inline float get_pixel_size() { return GET_SETTING(PixelSize); }

inline void set_pixel_size(float value)
{
    // std::lock_guard<std::mutex> lock(mutex_);
    UPDATE_SETTING(PixelSize, value);
}

inline SpaceTransformation get_space_transformation() { return GET_SETTING(SpaceTransformation); }

inline TimeTransformation get_time_transformation() { return GET_SETTING(TimeTransformation); }

inline ImgType get_img_type() { return GET_SETTING(ImageType); }
inline void set_img_type(ImgType type) { UPDATE_SETTING(ImageType, type); }

inline uint get_input_buffer_size() { return static_cast<uint>(GET_SETTING(InputBufferSize)); }
inline void set_input_buffer_size(uint value) { UPDATE_SETTING(InputBufferSize, value); }

inline uint get_time_stride() { return GET_SETTING(TimeStride); }
inline void set_time_stride(uint value)
{
    UPDATE_SETTING(TimeStride, value);

    uint batch_size = GET_SETTING(BatchSize);

    if (batch_size > value)
        UPDATE_SETTING(TimeStride, batch_size);
    // Go to lower multiple
    if (value % batch_size != 0)
        UPDATE_SETTING(TimeStride, value - value % batch_size);
}

inline uint get_batch_size() { return GET_SETTING(BatchSize); }
inline bool set_batch_size(uint value)
{
    bool request_time_stride_update = false;
    UPDATE_SETTING(BatchSize, value);

    if (value > get_input_buffer_size())
        value = get_input_buffer_size();
    uint time_stride = get_time_stride();
    if (time_stride < value)
    {
        UPDATE_SETTING(TimeStride, value);
        time_stride = value;
        request_time_stride_update = true;
    }
    // Go to lower multiple
    if (time_stride % value != 0)
        set_time_stride(time_stride - time_stride % value);

    return request_time_stride_update;
}

inline uint get_time_transformation_size() { return GET_SETTING(TimeTransformationSize); }

inline void set_time_transformation_size(uint value) { UPDATE_SETTING(TimeTransformationSize, value); }

inline float get_lambda() { return GET_SETTING(Lambda); }

inline float get_z_distance() { return GET_SETTING(ZDistance); }

inline std::vector<float> get_convo_matrix() { return GET_SETTING(ConvolutionMatrix); };

inline void set_convo_matrix(std::vector<float> value) { UPDATE_SETTING(ConvolutionMatrix, value); }

inline float get_contrast_lower_threshold() { return GET_SETTING(ContrastLowerThreshold); }
inline void set_contrast_lower_threshold(float value) { UPDATE_SETTING(ContrastLowerThreshold, value); }

inline float get_contrast_upper_threshold() { return GET_SETTING(ContrastUpperThreshold); }
inline void set_contrast_upper_threshold(float value) { UPDATE_SETTING(ContrastUpperThreshold, value); }

inline uint get_cuts_contrast_p_offset() { return static_cast<uint>(GET_SETTING(CutsContrastPOffset)); }
inline void set_cuts_contrast_p_offset(uint value) { UPDATE_SETTING(CutsContrastPOffset, value); }

inline unsigned get_renorm_constant() { return GET_SETTING(RenormConstant); }
inline void set_renorm_constant(unsigned int value) { UPDATE_SETTING(RenormConstant, value); }

inline int get_filter2d_n1() { return GET_SETTING(Filter2dN1); }
inline void set_filter2d_n1(int value)
{
    UPDATE_SETTING(Filter2dN1, value);
    set_auto_contrast_all();
}

inline int get_filter2d_n2() { return GET_SETTING(Filter2dN2); }
inline void set_filter2d_n2(int value)
{
    UPDATE_SETTING(Filter2dN2, value);
    set_auto_contrast_all();
}

inline int get_filter2d_smooth_low() { return GET_SETTING(Filter2dSmoothLow); }
inline void set_filter2d_smooth_low(int value) { UPDATE_SETTING(Filter2dSmoothLow, value); }

inline int get_filter2d_smooth_high() { return GET_SETTING(Filter2dSmoothHigh); }
inline void set_filter2d_smooth_high(int value) { UPDATE_SETTING(Filter2dSmoothHigh, value); }

inline float get_display_rate() { return GET_SETTING(DisplayRate); }
inline void set_display_rate(float value) { UPDATE_SETTING(DisplayRate, value); }
inline holovibes::Device get_input_queue_location() { return GET_SETTING(InputQueueLocation); }
inline void set_input_queue_location(holovibes::Device value) { UPDATE_SETTING(InputQueueLocation, value); }

inline ViewXY get_x(void) { return GET_SETTING(X); }

inline uint get_x_cuts() { return GET_SETTING(X).start; }

inline int get_x_accu_level() { return GET_SETTING(X).width; }

inline ViewXY get_y(void) { return GET_SETTING(Y); }

inline uint get_y_cuts() { return GET_SETTING(Y).start; }

inline int get_y_accu_level() { return GET_SETTING(Y).width; }

// XY
inline ViewXYZ get_xy() { return GET_SETTING(XY); }
inline bool get_xy_horizontal_flip() { return GET_SETTING(XY).horizontal_flip; }
inline float get_xy_rotation() { return GET_SETTING(XY).rotation; }
inline uint get_xy_accumulation_level() { return GET_SETTING(XY).output_image_accumulation; }
inline bool get_xy_enabled() { return GET_SETTING(XY).enabled; }
inline bool get_xy_log_enabled() { return GET_SETTING(XY).log_enabled; }
inline bool get_xy_contrast_enabled() { return GET_SETTING(XY).contrast.enabled; }
inline bool get_xy_contrast_auto_refresh() { return GET_SETTING(XY).contrast.auto_refresh; }
inline bool get_xy_contrast_invert() { return GET_SETTING(XY).contrast.invert; }
inline float get_xy_contrast_min() { return GET_SETTING(XY).contrast.min; }
inline float get_xy_contrast_max() { return GET_SETTING(XY).contrast.max; }
inline bool get_xy_img_accu_enabled() { return GET_SETTING(XY).output_image_accumulation > 1; }

inline void set_xy(ViewXYZ value) noexcept { UPDATE_SETTING(XY, value); }

inline void set_xy_horizontal_flip(bool value) noexcept
{
    auto xy = GET_SETTING(XY);
    xy.horizontal_flip = value;
    UPDATE_SETTING(XY, xy);
}

inline void set_xy_rotation(float value) noexcept
{
    auto xy = GET_SETTING(XY);
    xy.rotation = value;
    UPDATE_SETTING(XY, xy);
}

inline void set_xy_accumulation_level(uint value)
{
    auto xy = GET_SETTING(XY);
    xy.output_image_accumulation = value;
    UPDATE_SETTING(XY, xy);
}

inline void set_xy_log_enabled(bool value) noexcept
{
    auto xy = GET_SETTING(XY);
    xy.log_enabled = value;
    UPDATE_SETTING(XY, xy);
}

inline void set_xy_contrast_enabled(bool value) noexcept
{
    auto xy = GET_SETTING(XY);
    xy.contrast.enabled = value;
    UPDATE_SETTING(XY, xy);
    pipe_refresh();
}

inline void set_xy_contrast_auto_refresh(bool value) noexcept
{
    auto xy = GET_SETTING(XY);
    xy.contrast.auto_refresh = value;
    UPDATE_SETTING(XY, xy);
}

inline void set_xy_contrast_invert(bool value) noexcept
{
    auto xy = GET_SETTING(XY);
    xy.contrast.invert = value;
    UPDATE_SETTING(XY, xy);
}

inline void set_xy_contrast_min(float value) noexcept
{
    auto xy = GET_SETTING(XY);
    xy.contrast.min = value > 1.0f ? value : 1.0f;
    UPDATE_SETTING(XY, xy);
}

inline void set_xy_contrast_max(float value) noexcept
{
    auto xy = GET_SETTING(XY);
    xy.contrast.max = value > 1.0f ? value : 1.0f;
    UPDATE_SETTING(XY, xy);
}

inline void set_xy_contrast(float min, float max) noexcept
{
    auto xy = GET_SETTING(XY);
    xy.contrast.min = min;
    xy.contrast.max = max;
    UPDATE_SETTING(XY, xy);
}

// XZ
inline ViewXYZ get_xz() { return GET_SETTING(XZ); }
inline bool get_xz_enabled() { return GET_SETTING(XZ).enabled; }
inline bool get_xz_horizontal_flip() { return GET_SETTING(XZ).horizontal_flip; }
inline float get_xz_rotation() { return GET_SETTING(XZ).rotation; }
inline uint get_xz_accumulation_level() { return GET_SETTING(XZ).output_image_accumulation; }
inline bool get_xz_log_enabled() { return GET_SETTING(XZ).log_enabled; }
inline bool get_xz_contrast_enabled() { return GET_SETTING(XZ).contrast.enabled; }
inline bool get_xz_contrast_auto_refresh() { return GET_SETTING(XZ).contrast.auto_refresh; }
inline bool get_xz_contrast_invert() { return GET_SETTING(XZ).contrast.invert; }
inline float get_xz_contrast_min() { return GET_SETTING(XZ).contrast.min; }
inline float get_xz_contrast_max() { return GET_SETTING(XZ).contrast.max; }
inline bool get_xz_img_accu_enabled() { return GET_SETTING(XZ).output_image_accumulation > 1; }
inline void set_xz(ViewXYZ value) noexcept { UPDATE_SETTING(XZ, value); }
inline void set_xz_enabled(bool value) noexcept
{
    auto xz = GET_SETTING(XZ);
    xz.enabled = value;
    UPDATE_SETTING(XZ, xz);
}

inline void set_xz_horizontal_flip(bool value) noexcept
{
    auto xz = GET_SETTING(XZ);
    xz.horizontal_flip = value;
    UPDATE_SETTING(XZ, xz);
}

inline void set_xz_rotation(float value) noexcept
{
    auto xz = GET_SETTING(XZ);
    xz.rotation = value;
    UPDATE_SETTING(XZ, xz);
}

inline void set_xz_accumulation_level(uint value)
{
    auto xz = GET_SETTING(XZ);
    xz.output_image_accumulation = value;
    UPDATE_SETTING(XZ, xz);
}

inline void set_xz_log_enabled(bool value) noexcept
{
    auto xz = GET_SETTING(XZ);
    xz.log_enabled = value;
    UPDATE_SETTING(XZ, xz);
}

inline void set_xz_contrast_enabled(bool value) noexcept
{
    auto xz = GET_SETTING(XZ);
    xz.contrast.enabled = value;
    UPDATE_SETTING(XZ, xz);
}

inline void set_xz_contrast_auto_refresh(bool value) noexcept
{
    auto xz = GET_SETTING(XZ);
    xz.contrast.auto_refresh = value;
    UPDATE_SETTING(XZ, xz);
}

inline holovibes::Device get_record_queue_location() { return GET_SETTING(RecordQueueLocation); }

inline void set_xz_contrast_invert(bool value) noexcept
{
    auto xz = GET_SETTING(XZ);
    xz.contrast.invert = value;
    UPDATE_SETTING(XZ, xz);
}

inline void set_xz_contrast_min(float value) noexcept
{
    auto xz = GET_SETTING(XZ);
    xz.contrast.min = value > 1.0f ? value : 1.0f;
    UPDATE_SETTING(XZ, xz);
}

inline void set_xz_contrast_max(float value) noexcept
{
    auto xz = GET_SETTING(XZ);
    xz.contrast.max = value > 1.0f ? value : 1.0f;
    UPDATE_SETTING(XZ, xz);
}

inline void set_xz_contrast(float min, float max) noexcept
{
    auto xz = GET_SETTING(XZ);
    xz.contrast.min = min;
    xz.contrast.max = max;
    UPDATE_SETTING(XZ, xz);
}

inline ViewXYZ get_yz() { return GET_SETTING(YZ); }
inline bool get_yz_enabled() { return GET_SETTING(YZ).enabled; }
inline bool get_yz_horizontal_flip() { return GET_SETTING(YZ).horizontal_flip; }
inline float get_yz_rotation() { return GET_SETTING(YZ).rotation; }
inline uint get_yz_accumulation_level() { return GET_SETTING(YZ).output_image_accumulation; }
inline bool get_yz_log_enabled() { return GET_SETTING(YZ).log_enabled; }
inline bool get_yz_contrast_enabled() { return GET_SETTING(YZ).contrast.enabled; }
inline bool get_yz_contrast_auto_refresh() { return GET_SETTING(YZ).contrast.auto_refresh; }
inline bool get_yz_contrast_invert() { return GET_SETTING(YZ).contrast.invert; }
inline float get_yz_contrast_min() { return GET_SETTING(YZ).contrast.min; }
inline float get_yz_contrast_max() { return GET_SETTING(YZ).contrast.max; }
inline bool get_yz_img_accu_enabled() { return GET_SETTING(YZ).output_image_accumulation > 1; }
inline void set_yz(ViewXYZ value) noexcept
{
    LOG_INFO("New YZ");
    UPDATE_SETTING(YZ, value);
}
inline void set_yz_enabled(bool value) noexcept
{
    auto yz = GET_SETTING(YZ);
    yz.enabled = value;
    UPDATE_SETTING(YZ, yz);
}

inline void set_yz_horizontal_flip(bool value) noexcept
{
    auto yz = GET_SETTING(YZ);
    yz.horizontal_flip = value;
    UPDATE_SETTING(YZ, yz);
}

inline void set_yz_rotation(float value) noexcept
{
    auto yz = GET_SETTING(YZ);
    yz.rotation = value;
    UPDATE_SETTING(YZ, yz);
}

inline void set_yz_accumulation_level(uint value)
{
    auto yz = GET_SETTING(YZ);
    yz.output_image_accumulation = value;
    UPDATE_SETTING(YZ, yz);
}

inline void set_yz_log_enabled(bool value) noexcept
{
    auto yz = GET_SETTING(YZ);
    yz.log_enabled = value;
    UPDATE_SETTING(YZ, yz);
}

inline void set_yz_contrast_enabled(bool value) noexcept
{
    auto yz = GET_SETTING(YZ);
    yz.contrast.enabled = value;
    UPDATE_SETTING(YZ, yz);
}

inline void set_yz_contrast_auto_refresh(bool value) noexcept
{
    auto yz = GET_SETTING(YZ);
    yz.contrast.auto_refresh = value;
    UPDATE_SETTING(YZ, yz);
}

inline void set_yz_contrast_invert(bool value) noexcept
{
    auto yz = GET_SETTING(YZ);
    yz.contrast.invert = value;
    UPDATE_SETTING(YZ, yz);
}

inline void set_yz_contrast_min(float value) noexcept
{
    auto yz = GET_SETTING(YZ);
    yz.contrast.min = value > 1.0f ? value : 1.0f;
    UPDATE_SETTING(YZ, yz);
}

inline void set_yz_contrast_max(float value) noexcept
{
    auto yz = GET_SETTING(YZ);
    yz.contrast.max = value > 1.0f ? value : 1.0f;
    UPDATE_SETTING(YZ, yz);
}

inline void set_yz_contrast(float min, float max) noexcept
{
    auto yz = GET_SETTING(YZ);
    yz.contrast.min = min;
    yz.contrast.max = max;
    UPDATE_SETTING(YZ, yz);
}

inline ViewWindow get_filter2d() { return GET_SETTING(Filter2d); }
inline bool get_filter2d_log_enabled() { return GET_SETTING(Filter2d).log_enabled; }
inline bool get_filter2d_contrast_enabled() { return GET_SETTING(Filter2d).contrast.enabled; }
inline bool get_filter2d_contrast_auto_refresh() { return GET_SETTING(Filter2d).contrast.auto_refresh; }
inline bool get_filter2d_contrast_invert() { return GET_SETTING(Filter2d).contrast.invert; }
inline float get_filter2d_contrast_min() { return GET_SETTING(Filter2d).contrast.min; }
inline float get_filter2d_contrast_max() { return GET_SETTING(Filter2d).contrast.max; }

inline void set_filter2d(ViewWindow value) noexcept { UPDATE_SETTING(Filter2d, value); }

inline void set_filter2d_log_enabled(bool value) noexcept
{
    auto filter2d = GET_SETTING(Filter2d);
    filter2d.log_enabled = value;
    UPDATE_SETTING(Filter2d, filter2d);
}

inline void set_filter2d_contrast_enabled(bool value) noexcept
{
    auto filter2d = GET_SETTING(Filter2d);
    filter2d.contrast.enabled = value;
    UPDATE_SETTING(Filter2d, filter2d);
}

inline void set_filter2d_contrast_auto_refresh(bool value) noexcept
{
    auto filter2d = GET_SETTING(Filter2d);
    filter2d.contrast.auto_refresh = value;
    UPDATE_SETTING(Filter2d, filter2d);
}

inline void set_filter2d_contrast_invert(bool value) noexcept
{
    auto filter2d = GET_SETTING(Filter2d);
    filter2d.contrast.invert = value;
    UPDATE_SETTING(Filter2d, filter2d);
}

inline void set_filter2d_contrast_min(float value) noexcept
{
    auto filter2d = GET_SETTING(Filter2d);
    filter2d.contrast.min = value > 1.0f ? value : 1.0f;
    UPDATE_SETTING(Filter2d, filter2d);
}

inline void set_filter2d_contrast_max(float value) noexcept
{
    auto filter2d = GET_SETTING(Filter2d);
    filter2d.contrast.max = value > 1.0f ? value : 1.0f;
    UPDATE_SETTING(Filter2d, filter2d);
}

inline void set_filter2d_contrast(float min, float max) noexcept
{
    auto filter2d = GET_SETTING(Filter2d);
    filter2d.contrast.min = min;
    filter2d.contrast.max = max;
    UPDATE_SETTING(Filter2d, filter2d);
}

inline float get_reticle_scale() { return GET_SETTING(ReticleScale); }
inline void set_reticle_scale(float value) { UPDATE_SETTING(ReticleScale, value); }

inline int get_unwrap_history_size() { return GET_SETTING(UnwrapHistorySize); }

inline bool get_is_computation_stopped() { return GET_SETTING(IsComputationStopped); }
inline void set_is_computation_stopped(bool value) { UPDATE_SETTING(IsComputationStopped, value); }

inline bool get_divide_convolution_enabled() { return GET_SETTING(DivideConvolutionEnabled); }
inline void set_divide_convolution_enabled(bool value) { UPDATE_SETTING(DivideConvolutionEnabled, value); }

inline bool get_renorm_enabled() { return GET_SETTING(RenormEnabled); }
inline void set_renorm_enabled(bool value) { UPDATE_SETTING(RenormEnabled, value); }

inline bool get_fft_shift_enabled() { return GET_SETTING(FftShiftEnabled); }

inline void set_fft_shift_enabled(bool value)
{
    UPDATE_SETTING(FftShiftEnabled, value);
    pipe_refresh();
}

inline bool get_raw_view_enabled() { return GET_SETTING(RawViewEnabled); }

inline holovibes::Device get_raw_view_queue_location() { return GET_SETTING(RawViewQueueLocation); }

inline void set_raw_view_queue_location(holovibes::Device value) { UPDATE_SETTING(RawViewQueueLocation, value); }

inline void set_raw_view_enabled(bool value) { UPDATE_SETTING(RawViewEnabled, value); }

inline bool get_filter2d_enabled() { return GET_SETTING(Filter2dEnabled); }
inline void set_filter2d_enabled(bool value) { UPDATE_SETTING(Filter2dEnabled, value); }

inline bool get_filter2d_view_enabled() { return GET_SETTING(Filter2dViewEnabled); }

inline bool get_cuts_view_enabled() { return GET_SETTING(CutsViewEnabled); }
inline void set_cuts_view_enabled(bool value) { UPDATE_SETTING(CutsViewEnabled, value); }

inline bool get_lens_view_enabled() { return GET_SETTING(LensViewEnabled); }
inline void set_lens_view_enabled(bool value) { UPDATE_SETTING(LensViewEnabled, value); }

inline bool get_chart_display_enabled() { return GET_SETTING(ChartDisplayEnabled); }

inline bool get_reticle_display_enabled() { return GET_SETTING(ReticleDisplayEnabled); }
inline void set_reticle_display_enabled(bool value) { UPDATE_SETTING(ReticleDisplayEnabled, value); }

inline uint get_file_buffer_size() { return static_cast<uint>(GET_SETTING(FileBufferSize)); }

inline void set_file_buffer_size(uint value) { UPDATE_SETTING(FileBufferSize, value); }

inline uint get_record_buffer_size() { return static_cast<uint>(GET_SETTING(RecordBufferSize)); }

inline uint get_time_transformation_cuts_output_buffer_size()
{
    return GET_SETTING(TimeTransformationCutsOutputBufferSize);
}
inline void set_time_transformation_cuts_output_buffer_size(uint value)
{
    UPDATE_SETTING(TimeTransformationCutsOutputBufferSize, value);
}

inline bool get_batch_enabled() { return GET_SETTING(BatchEnabled); }

inline void set_batch_enabled(bool value) { UPDATE_SETTING(BatchEnabled, value); }

inline std::optional<std::string> get_batch_file_path() { return GET_SETTING(BatchFilePath); }

inline void set_batch_file_path(std::string value) { UPDATE_SETTING(BatchFilePath, value); }

inline std::string get_record_file_path() { return GET_SETTING(RecordFilePath); }

inline void set_record_file_path(std::string value) { UPDATE_SETTING(RecordFilePath, value); }

inline std::optional<size_t> get_record_frame_count() { return GET_SETTING(RecordFrameCount); }

inline void set_record_frame_count(std::optional<size_t> value) { UPDATE_SETTING(RecordFrameCount, value); }

inline RecordMode get_record_mode() { return GET_SETTING(RecordMode); }

inline void set_record_mode(RecordMode value) { UPDATE_SETTING(RecordMode, value); }

inline void set_benchmark_mode(bool value) { UPDATE_SETTING(BenchmarkMode, value); }

inline bool get_benchmark_mode() { return GET_SETTING(BenchmarkMode); }

inline void set_record_on_gpu(bool value) { UPDATE_SETTING(RecordOnGPU, value); }

inline bool get_record_on_gpu() { return GET_SETTING(RecordOnGPU); }

inline size_t get_record_frame_skip() { return GET_SETTING(RecordFrameSkip); }

inline void set_record_frame_skip(size_t value) { UPDATE_SETTING(RecordFrameSkip, value); }

inline size_t get_output_buffer_size() { return GET_SETTING(OutputBufferSize); }

inline void set_output_buffer_size(size_t value) { UPDATE_SETTING(OutputBufferSize, value); }

inline uint get_input_fps() { return static_cast<uint>(GET_SETTING(InputFPS)); }

inline void set_input_fps(uint value) { UPDATE_SETTING(InputFPS, value); }

inline std::string get_input_file_path() { return GET_SETTING(InputFilePath); }

inline void set_input_file_path(std::string value) { UPDATE_SETTING(InputFilePath, value); }

inline bool get_loop_on_input_file() { return GET_SETTING(LoopOnInputFile); }

inline void set_loop_on_input_file(bool value) { UPDATE_SETTING(LoopOnInputFile, value); }

inline bool get_load_file_in_gpu() { return GET_SETTING(LoadFileInGPU); }

inline void set_load_file_in_gpu(bool value) { UPDATE_SETTING(LoadFileInGPU, value); }

inline size_t get_input_file_start_index() { return GET_SETTING(InputFileStartIndex); }

inline size_t get_input_file_end_index() { return GET_SETTING(InputFileEndIndex); }

inline ViewPQ get_p() { return GET_SETTING(P); }

inline int get_p_accu_level() { return GET_SETTING(P).width; }

inline uint get_p_index() { return GET_SETTING(P).start; }

inline ViewPQ get_q(void) { return GET_SETTING(Q); }

inline uint get_q_index() { return GET_SETTING(Q).start; }

inline uint get_q_accu_level() { return GET_SETTING(Q).width; }

inline const camera::FrameDescriptor& get_fd() { return Holovibes::instance().get_input_queue()->get_fd(); };

inline std::shared_ptr<Pipe> get_compute_pipe() { return Holovibes::instance().get_compute_pipe(); };
inline std::shared_ptr<Pipe> get_compute_pipe_no_throw() { return Holovibes::instance().get_compute_pipe_no_throw(); };

inline std::shared_ptr<Queue> get_gpu_output_queue() { return Holovibes::instance().get_gpu_output_queue(); };

inline std::shared_ptr<BatchInputQueue> get_input_queue() { return Holovibes::instance().get_input_queue(); };

inline units::RectFd get_signal_zone() { return GET_SETTING(SignalZone); };
inline units::RectFd get_noise_zone() { return GET_SETTING(NoiseZone); };
inline units::RectFd get_composite_zone() { return GET_SETTING(CompositeZone); };
inline units::RectFd get_zoomed_zone() { return GET_SETTING(ZoomedZone); };
inline units::RectFd get_reticle_zone() { return GET_SETTING(ReticleZone); };

inline void set_signal_zone(const units::RectFd& rect) { UPDATE_SETTING(SignalZone, rect); };
inline void set_noise_zone(const units::RectFd& rect) { UPDATE_SETTING(NoiseZone, rect); };
inline void set_composite_zone(const units::RectFd& rect) { UPDATE_SETTING(CompositeZone, rect); };
inline void set_zoomed_zone(const units::RectFd& rect) { UPDATE_SETTING(ZoomedZone, rect); };
inline void set_reticle_zone(const units::RectFd& rect) { UPDATE_SETTING(ReticleZone, rect); };

inline void set_frame_record_enabled(bool value) { UPDATE_SETTING(FrameRecordEnabled, value); }

inline bool get_frame_record_enabled() { return GET_SETTING(FrameRecordEnabled); }

inline void set_chart_record_enabled(bool value) { UPDATE_SETTING(ChartRecordEnabled, value); }

inline bool get_chart_record_enabled() { return GET_SETTING(ChartRecordEnabled); }

inline void set_convolution_enabled(bool value) { UPDATE_SETTING(ConvolutionEnabled, value); }

inline bool get_convolution_enabled() { return GET_SETTING(ConvolutionEnabled); }

inline void set_filter_enabled(bool value) { UPDATE_SETTING(FilterEnabled, value); };

inline bool get_filter_enabled() { return GET_SETTING(FilterEnabled); };

// Ex composite_cache
inline CompositeKind get_composite_kind() noexcept { return GET_SETTING(CompositeKind); }

inline bool get_composite_auto_weights() noexcept { return GET_SETTING(CompositeAutoWeights); }

inline bool get_z_fft_shift() noexcept { return GET_SETTING(ZFFTShift); }

// RGB
inline CompositeRGB get_rgb() noexcept { return GET_SETTING(RGB); }
inline uint get_rgb_p_min() noexcept { return GET_SETTING(RGB).frame_index.min; }
inline uint get_rgb_p_max() noexcept { return GET_SETTING(RGB).frame_index.max; }
inline float get_weight_r() noexcept { return GET_SETTING(RGB).weight.r; }
inline float get_weight_g() noexcept { return GET_SETTING(RGB).weight.g; }
inline float get_weight_b() noexcept { return GET_SETTING(RGB).weight.b; }

// HSV
inline CompositeHSV get_hsv() noexcept { return GET_SETTING(HSV); }
inline uint get_composite_p_min_h() noexcept { return GET_SETTING(HSV).h.frame_index.min; }
inline uint get_composite_p_max_h() noexcept { return GET_SETTING(HSV).h.frame_index.max; }

inline float get_slider_h_threshold_min() noexcept { return GET_SETTING(HSV).h.slider_threshold.min; }
inline float get_slider_h_threshold_max() noexcept { return GET_SETTING(HSV).h.slider_threshold.max; }

inline float get_slider_h_shift_min() { return GET_SETTING(HSV).h.slider_shift.min; }

inline float get_slider_h_shift_max() { return GET_SETTING(HSV).h.slider_shift.max; }

inline float get_composite_low_h_threshold() noexcept { return GET_SETTING(HSV).h.threshold.min; }
inline float get_composite_high_h_threshold() noexcept { return GET_SETTING(HSV).h.threshold.max; }
inline uint get_composite_p_min_s() noexcept { return GET_SETTING(HSV).s.frame_index.min; }
inline uint get_composite_p_max_s() noexcept { return GET_SETTING(HSV).s.frame_index.max; }
inline float get_slider_s_threshold_min() noexcept { return GET_SETTING(HSV).s.slider_threshold.min; }
inline float get_slider_s_threshold_max() noexcept { return GET_SETTING(HSV).s.slider_threshold.max; }
inline float get_composite_low_s_threshold() noexcept { return GET_SETTING(HSV).s.threshold.min; }
inline float get_composite_high_s_threshold() noexcept { return GET_SETTING(HSV).s.threshold.max; }
inline uint get_composite_p_min_v() noexcept { return GET_SETTING(HSV).v.frame_index.min; }
inline uint get_composite_p_max_v() noexcept { return GET_SETTING(HSV).v.frame_index.max; }
inline float get_slider_v_threshold_min() noexcept { return GET_SETTING(HSV).v.slider_threshold.min; }
inline float get_slider_v_threshold_max() noexcept { return GET_SETTING(HSV).v.slider_threshold.max; }
inline float get_composite_low_v_threshold() noexcept { return GET_SETTING(HSV).v.threshold.min; }
inline float get_composite_high_v_threshold() noexcept { return GET_SETTING(HSV).v.threshold.max; }
inline bool get_composite_p_activated_s() noexcept { return GET_SETTING(HSV).s.frame_index.activated; }
inline bool get_composite_p_activated_v() noexcept { return GET_SETTING(HSV).v.frame_index.activated; }
inline uint get_composite_p_red() { return GET_SETTING(RGB).frame_index.min; }
inline uint get_composite_p_blue() { return GET_SETTING(RGB).frame_index.max; }

inline void set_composite_kind(CompositeKind value) { UPDATE_SETTING(CompositeKind, value); }

inline void set_composite_auto_weights(bool value)
{
    UPDATE_SETTING(CompositeAutoWeights, value);
    pipe_refresh();
}

// RGB
inline void set_rgb(CompositeRGB value) { UPDATE_SETTING(RGB, value); }

inline void set_rgb_p(int min, int max)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.frame_index.min = min;
    rgb.frame_index.max = max;
    UPDATE_SETTING(RGB, rgb);
}

inline void set_weight_r(double value)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.weight.r = value;
    UPDATE_SETTING(RGB, rgb);
}
inline void set_weight_g(double value)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.weight.g = value;
    UPDATE_SETTING(RGB, rgb);
}
inline void set_weight_b(double value)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.weight.b = value;
    UPDATE_SETTING(RGB, rgb);
}

inline void set_weight_rgb(double r, double g, double b)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.weight.r = r;
    rgb.weight.g = g;
    rgb.weight.b = b;
    UPDATE_SETTING(RGB, rgb);
}

// HSV

inline void set_z_fft_shift(bool checked) { UPDATE_SETTING(ZFFTShift, checked); }

inline void set_composite_p_h(int min, int max)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.frame_index.min = min;
    hsv.h.frame_index.max = max;
    UPDATE_SETTING(HSV, hsv);
}

inline void set_hsv(CompositeHSV value) { UPDATE_SETTING(HSV, value); }
inline void set_slider_h_threshold_min(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.slider_threshold.min = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_slider_h_threshold_max(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.slider_threshold.max = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_slider_h_shift_min(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.slider_shift.min = value;
    UPDATE_SETTING(HSV, hsv);
}

inline void set_slider_h_shift_max(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.slider_shift.max = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_low_h_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.threshold.min = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_high_h_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.threshold.max = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_p_min_h(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.frame_index.min = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_p_max_h(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.frame_index.max = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_p_min_s(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.frame_index.min = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_p_max_s(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.frame_index.max = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_slider_s_threshold_min(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.slider_threshold.min = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_slider_s_threshold_max(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.slider_threshold.max = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_low_s_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.threshold.min = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_high_s_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.threshold.max = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_p_min_v(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.frame_index.min = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_p_max_v(uint value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.frame_index.max = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_slider_v_threshold_min(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.slider_threshold.min = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_slider_v_threshold_max(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.slider_threshold.max = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_low_v_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.threshold.min = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_high_v_threshold(float value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.threshold.max = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_p_activated_s(bool value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.s.frame_index.activated = value;
    UPDATE_SETTING(HSV, hsv);
}
inline void set_composite_p_activated_v(bool value)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.v.frame_index.activated = value;
    UPDATE_SETTING(HSV, hsv);
}

inline std::optional<size_t> get_nb_frames_to_record() { return GET_SETTING(RecordFrameCount); }
inline void set_nb_frames_to_record(std::optional<size_t> nb_frames) { UPDATE_SETTING(RecordFrameCount, nb_frames); }
inline uint get_nb_frame_skip() { return GET_SETTING(FrameSkip); }
inline uint get_mp4_fps() { return GET_SETTING(Mp4Fps); }

#pragma endregion

} // namespace holovibes::api
