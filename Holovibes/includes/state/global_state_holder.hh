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
#include "all_caches.hh"
#include "cache_dispatcher.hh"
#include "all_pipe_requests_on_sync_functions.hh"
#include "gsh_cache_on_change.hh"
#include "compute_gsh_on_change.hh"

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
 * have impact on buffers'size would have caused incoherent ComputeModeEnums and/or segfaults and undefined behaviors).
 * The ComputeWorker now possess MicroCaches, containing all the state it needs. Those MicroCaches are accessed only by
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

// clang-format off
class GSHAdvancedCache : public AdvancedCache::Ref<>{};
class GSHComputeCache : public ComputeCache::Ref<ComputeGSHOnChange>{};
class GSHImportCache : public ImportCache::Ref<ImportGSHOnChange>{};
class GSHExportCache : public ExportCache::Ref<>{};
class GSHCompositeCache : public CompositeCache::Ref<>{};
class GSHViewCache : public ViewCache::Ref<ViewGSHOnChange>{};
class GSHZoneCache : public ZoneCache::Ref<>{};
// clang-format on

using GSHCacheDispatcher = CacheDispatcher<GSHAdvancedCache,
                                           GSHComputeCache,
                                           GSHImportCache,
                                           GSHExportCache,
                                           GSHCompositeCache,
                                           GSHViewCache,
                                           GSHZoneCache>;

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
    typename T::ConstRefType get_value()
    {
        return cache_dispatcher_.template get<T>().template get_value<T>();
    }

    template <typename T>
    void set_value(typename T::ConstRefType value)
    {
        return cache_dispatcher_.template get<T>().template set_value<T>(value);
    }

    template <typename T>
    TriggerChangeValue<typename T::ValueType> change_value()
    {
        return cache_dispatcher_.template get<T>().template change_value<T>();
    }

    auto& get_advanced_cache() { return advanced_cache_; }
    auto& get_compute_cache() { return compute_cache_; }
    auto& get_import_cache() { return import_cache_; }
    auto& get_export_cache() { return export_cache_; }
    auto& get_composite_cache() { return composite_cache_; }
    auto& get_view_cache() { return view_cache_; }
    auto& get_zone_cache() { return zone_cache_; }

  public:
    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

  public:
    enum class ComputeSettingsVersion
    {
        V2,
        V3,
        V4,
        V5
    };
    static void convert_json(json& data, GSH::ComputeSettingsVersion from);

    void set_notify_callback(std::function<void()> func) { notify_callback_ = func; }
    void notify() { notify_callback_(); }

    void set_caches_as_refs();
    void remove_caches_as_refs();

  private:
    std::function<void()> notify_callback_ = []() {};

    GSHAdvancedCache advanced_cache_;
    GSHComputeCache compute_cache_;
    GSHImportCache import_cache_;
    GSHExportCache export_cache_;
    GSHCompositeCache composite_cache_;
    GSHViewCache view_cache_;
    GSHZoneCache zone_cache_;

    GSHCacheDispatcher cache_dispatcher_;

    // FIXME : HOOW
    mutable std::mutex mutex_;
};

} // namespace holovibes
