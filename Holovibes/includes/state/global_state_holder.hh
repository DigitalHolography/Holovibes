/*! \file
 *
 * \brief Global State Holder Class
 *
 *  Holds the state of the entire application
 */

#pragma once

#include <mutex>

#include "fast_updates_holder.hh"
#include "caches.hh"
#include "entities.hh"
#include "view_struct.hh"
#include "rendering_struct.hh"
#include "composite_struct.hh"
#include "internals_struct.hh"
#include "advanced_struct.hh"

namespace holovibes
{

using entities::Span;

/*! \class GSH
 *
 * \brief The GSH (global state holder), is where all the global state of the program is stored.
 *
 * Its goal is to register changes commanded by the API (which comes from user events), dispatch the changes
 * to each worker, and provide informations queried by the API. It relies on several structures and mecanisms :
 *
 * Queries and Commands : every data needed by the API shall be obtained in the form of structured Queries, provided by
 * the GSH. This guarantees that only needed data is accessed to, and grants better code readability.
 * The same principle is applied to changes comming from the API, which come in the form of structured Commands.
 *
 * MicroCache : local state holder belonging to a worker. Previously, each worker had to fetch data at the same
 * place ; therefore, all variables had to be atomic, with the aim to be thread safe. Furthermore, since global state
 * was used in the pipe, directly modifying the state was often not possible (changing operations or variables which
 * have impact on buffers'size would have caused incoherent computations and/or segfaults and undefined behaviors). The
 * ComputeWorker now possess MicroCaches, containing all the state it needs. Those MicroCaches are accessed only by
 * their worker, and are synchronized with the GSH when each worker chooses to (using a trigger system). The
 * implementation of MicroCaches enabled the direct modification of the state, since the state used in the pipe is now
 * desynchronized from the GSH.
 * More informations and explanations concerning their synchronization with the GSH are provided in files micro-cache.hh
 * and micro-cache.hxx.
 *
 * FastUpdateHolder : the fastUpdateHolder is a templated map which is used by the informationWorker to access and
 * display information (like fps and queue occupancy) at a high rate, since this needs to be updated continuously.
 */
class GSH
{
    static GSH* instance_;

  public:
    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

    // static inline GSH& instance() { return *instance_; }
    static GSH& instance();

    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

#pragma region(collapsed) GETTERS

    bool get_contrast_auto_refresh() const noexcept;
    bool get_contrast_invert() const noexcept;
    bool get_contrast_enabled() const noexcept;

    bool is_current_window_xyz_type() const;

    // Over current window
    float get_contrast_min() const;
    float get_contrast_max() const;
    double get_rotation() const;
    bool get_horizontal_flip() const;
    bool get_log_enabled() const;
    unsigned get_accumulation_level() const;

#pragma endregion

#pragma region(collapsed) SETTERS

    void set_batch_size(uint value);
    void set_time_transformation_size(uint value);
    void set_time_stride(uint value);

    void disable_convolution();
    void enable_convolution(std::optional<std::string> file);
    // Over current window
    void set_contrast_enabled(bool contrast_enabled);
    void set_contrast_auto_refresh(bool contrast_auto_refresh);
    void set_contrast_invert(bool contrast_invert);
    void set_contrast_min(float value);
    void set_contrast_max(float value);
    void set_log_enabled(bool value);
    void set_accumulation_level(int value);
    void set_rotation(double value);
    void set_horizontal_flip(double value);

    void set_rgb_p();
    void set_weight_rgb();
    void set_composite_p_h();

    enum class ComputeSettingsVersion
    {
        V2,
        V3,
        V4,
        V5
    };
    static void convert_json(json& data, GSH::ComputeSettingsVersion from);

#pragma endregion

    void set_notify_callback(std::function<void()> func) { notify_callback_ = func; }

    void update_contrast(WindowKind kind, float min, float max);

    static void load_input_filter(std::vector<float> input_filter, const std::string& file);

  private:
    GSH() noexcept {}

    std::shared_ptr<holovibes::ViewWindow> get_window(WindowKind kind);

    std::function<void()> notify_callback_ = []() {};
    void notify() { notify_callback_(); }

    mutable std::mutex mutex_;
};

} // namespace holovibes
