/*! \file
 *
 * \brief Global State Holder Class
 *
 *  Holds the state of the entire application
 */

#pragma once

#include <mutex>

#include "fast_updates_holder.hh"
#include "entities.hh"
#include "view_struct.hh"
#include "rendering_struct.hh"
#include "composite_struct.hh"
#include "internals_struct.hh"
#include "advanced_struct.hh"
#include "gsh_parameters_handler.hh"
#include "cache_gsh.hh"

#include "all_caches.hh"
#include "cache_dispatcher.hh"

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

//! technically useless, but it's a great plus in order to don't take care of witch cache we refering to
using GSHCacheDispatcher = CacheDispatcher<AdvancedCache::Ref,
                                           ComputeCache::Ref,
                                           ExportCache::Ref,
                                           CompositeCache::Ref,
                                           Filter2DCache::Ref,
                                           ViewCache::Ref,
                                           ZoneCache::Ref,
                                           ImportCache::Ref,
                                           FileReadCache::Ref>;

class GSH
{
  private:
    GSH();
    ~GSH();

  public:
    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

  public:
    static GSH& instance();

  public:
    template <typename T>
    const typename T::RefType get_value()
    {
        return cache_dispatcher_.template get<T>().template get_value<T>();
    }

    template <typename T>
    void set_value(typename T::RefType value)
    {
        return cache_dispatcher_.template get<T>().template set_value<T>(value);
    }

    template <typename T>
    TriggerChangeValue<typename T::ValueType> change_value()
    {
        return cache_dispatcher_.template get<T>().template change_value<T>();
    }

    AdvancedCache::Ref& get_advanced_cache() { return advanced_cache_; }
    ComputeCache::Ref& get_compute_cache() { return compute_cache_; }
    ExportCache::Ref& get_export_cache() { return export_cache_; }
    CompositeCache::Ref& get_composite_cache() { return composite_cache_; }
    Filter2DCache::Ref& get_filter2d_cache() { return filter2d_cache_; }
    ViewCache::Ref& get_view_cache() { return view_cache_; }
    ZoneCache::Ref& get_zone_cache() { return zone_cache_; }
    ImportCache::Ref& get_import_cache() { return import_cache_; }
    FileReadCache::Ref& get_file_read_cache() { return file_read_cache_; }

  public:
    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

  private:
    View_Window* get_window_internal(WindowKind kind)
    {
        if (kind == WindowKind::XYview)
            return &view_cache_.get_value<ViewXY>();
        else if (kind == WindowKind::XZview)
            return &view_cache_.get_value<ViewXZ>();
        else if (kind == WindowKind::YZview)
            return &view_cache_.get_value<ViewYZ>();
        else if (kind == WindowKind::Filter2D)
            return &view_cache_.get_value<Filter2D>();

        throw std::runtime_error("Unexpected WindowKind");
        return nullptr;
    }

  public:

    enum class ComputeSettingsVersion
    {
        V2,
        V3,
        V4,
        V5
    };
    static void convert_json(json& data, GSH::ComputeSettingsVersion from);
      
    void change_window(WindowKind kind) { view_cache_.set_value<CurrentWindowKind>(kind); }

    View_Window& get_window(WindowKind kind) { return *get_window_internal(kind); }
    View_Window& get_current_window() { return *get_window_internal(view_cache_.get_value<CurrentWindowKind>()); }


    void set_notify_callback(std::function<void()> func) { notify_callback_ = func; }
    void notify() { notify_callback_(); }

    // FIXME
    void update_contrast(WindowKind kind, float min, float max)
    {
        View_Window& window = get_window(kind);
        window.set_contrast_min(min);
        window.set_contrast_max(max);

        notify();
    }

  private:
    std::function<void()> notify_callback_ = []() {};

    AdvancedCache::Ref advanced_cache_;
    ComputeCache::Ref compute_cache_;
    ExportCache::Ref export_cache_;
    CompositeCache::Ref composite_cache_;
    Filter2DCache::Ref filter2d_cache_;
    ViewCache::Ref view_cache_;
    ZoneCache::Ref zone_cache_;
    ImportCache::Ref import_cache_;
    FileReadCache::Ref file_read_cache_;

    GSHCacheDispatcher cache_dispatcher_;

    mutable std::mutex mutex_;
};

} // namespace holovibes
