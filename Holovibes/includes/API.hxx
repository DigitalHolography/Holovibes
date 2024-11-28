#pragma once

#include "API.hh"
#include "enum_record_mode.hh"
#include "enum_recorded_data_type.hh"

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

inline SpaceTransformation get_space_transformation() { return GET_SETTING(SpaceTransformation); }

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
    {
        request_time_stride_update = true;
        set_time_stride(time_stride - time_stride % value);
    }

    return request_time_stride_update;
}

inline float get_lambda() { return GET_SETTING(Lambda); }

inline float get_z_distance() { return GET_SETTING(ZDistance); }

inline bool get_benchmark_mode() { return GET_SETTING(BenchmarkMode); }
inline void set_benchmark_mode(bool value) { UPDATE_SETTING(BenchmarkMode, value); }

inline size_t get_output_buffer_size() { return GET_SETTING(OutputBufferSize); }
inline void set_output_buffer_size(size_t value) { UPDATE_SETTING(OutputBufferSize, value); }

inline const camera::FrameDescriptor& get_fd() { return Holovibes::instance().get_input_queue()->get_fd(); };

inline std::shared_ptr<Pipe> get_compute_pipe() { return Holovibes::instance().get_compute_pipe(); };
inline std::shared_ptr<Pipe> get_compute_pipe_no_throw() { return Holovibes::instance().get_compute_pipe_no_throw(); };

inline std::shared_ptr<Queue> get_gpu_output_queue() { return Holovibes::instance().get_gpu_output_queue(); };

inline std::shared_ptr<BatchInputQueue> get_input_queue() { return Holovibes::instance().get_input_queue(); };

inline bool get_is_computation_stopped() { return GET_SETTING(IsComputationStopped); }
inline void set_is_computation_stopped(bool value) { UPDATE_SETTING(IsComputationStopped, value); }

inline ViewPQ get_p() { return GET_SETTING(P); }
inline int get_p_accu_level() { return GET_SETTING(P).width; }
inline uint get_p_index() { return GET_SETTING(P).start; }

inline ViewPQ get_q(void) { return GET_SETTING(Q); }
inline uint get_q_index() { return GET_SETTING(Q).start; }
inline uint get_q_accu_level() { return GET_SETTING(Q).width; }

inline ViewXY get_x(void) { return GET_SETTING(X); }
inline uint get_x_cuts() { return GET_SETTING(X).start; }
inline int get_x_accu_level() { return GET_SETTING(X).width; }

inline ViewXY get_y(void) { return GET_SETTING(Y); }
inline uint get_y_cuts() { return GET_SETTING(Y).start; }
inline int get_y_accu_level() { return GET_SETTING(Y).width; }

/*!
 * \name Time transformation
 * \{
 */
inline TimeTransformation get_time_transformation() { return GET_SETTING(TimeTransformation); }

inline uint get_time_transformation_size() { return GET_SETTING(TimeTransformationSize); }
inline void set_time_transformation_size(uint value) { UPDATE_SETTING(TimeTransformationSize, value); }

inline uint get_time_transformation_cuts_output_buffer_size()
{
    return GET_SETTING(TimeTransformationCutsOutputBufferSize);
}
inline void set_time_transformation_cuts_output_buffer_size(uint value)
{
    UPDATE_SETTING(TimeTransformationCutsOutputBufferSize, value);
}
/*! \} */

/*!
 * \name FFT
 * \{
 */
/*! \brief Getter and Setter for the fft shift, triggered when FFT Shift button is clicked on the gui. (Setter refreshes
 * the pipe) */
inline bool get_fft_shift_enabled() { return GET_SETTING(FftShiftEnabled); }
inline bool get_registration_enabled();
inline void set_registration_enabled(bool value);
inline void set_fft_shift_enabled(bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    // Deactivate registration if fft shift is disabled
    if (api::get_registration_enabled())
        set_registration_enabled(value);

    UPDATE_SETTING(FftShiftEnabled, value);
    pipe_refresh();
}
/*! \} */

/*! \name Zone
 * \{
 */
inline units::RectFd get_signal_zone() { return GET_SETTING(SignalZone); };
inline units::RectFd get_noise_zone() { return GET_SETTING(NoiseZone); };
inline units::RectFd get_composite_zone() { return GET_SETTING(CompositeZone); };
inline units::RectFd get_zoomed_zone() { return GET_SETTING(ZoomedZone); };

inline void set_signal_zone(const units::RectFd& rect) { UPDATE_SETTING(SignalZone, rect); };
inline void set_noise_zone(const units::RectFd& rect) { UPDATE_SETTING(NoiseZone, rect); };
inline void set_composite_zone(const units::RectFd& rect) { UPDATE_SETTING(CompositeZone, rect); };
inline void set_zoomed_zone(const units::RectFd& rect) { UPDATE_SETTING(ZoomedZone, rect); };

    /*! \} */

#pragma endregion

} // namespace holovibes::api
