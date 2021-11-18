#pragma once

#include <mutex>

#include "fast_updates_holder.hh"
#include "caches.hh"
#include "entities.hh"

namespace holovibes
{

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
 * MicroCache : local state holder belonging to each worker. Previously, each worker had to fetch data at the same
 * place ; therefore, all variables had to be atomic, with the aim to be thread safe. In order to avoid that, each
 * worker now possess MicroCaches, containing all the state they need. Those MicroCaches are accessed only by their
 * worker, and are synchronized with the GSH when each worker chooses to (using a trigger system).
 * More informations and explanations concerning their synchronization with the GSH are provided in files micro-cache.hh
 * and micro-cache.hxx.
 *
 * FastUpdateHolder : the fastUpdateHolder is a templated map which is used by the informationWorker to access and
 * display information (like fps and queue occupancy) at a high rate, since this needs to be updated continuously.
 */
class GSH
{
  public:
    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

    static GSH& instance();

    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

    uint get_batch_size() const;
    uint get_time_transformation_size() const;
    uint get_time_transformation_stride() const;
    SpaceTransformation get_space_transformation() const;
    TimeTransformation get_time_transformation() const;
    int get_filter2d_n1() const;
    int get_filter2d_n2() const;

    void set_batch_size(uint value);
    void set_time_transformation_size(uint value);
    void set_time_transformation_stride(uint value);
    void set_space_transformation(const SpaceTransformation& value);
    void set_space_transformation_from_string(const std::string& value);
    void set_time_transformation(const TimeTransformation& value);
    void set_time_transformation_from_string(const std::string& value);
    void set_filter2d_n1(int value);
    void set_filter2d_n2(int value);

    void load_ptree(const boost::property_tree::ptree& ptree);

    void dump_ptree(boost::property_tree::ptree& ptree) const;

  private:
    GSH() {}

    ComputeCache::Ref compute_cache_;
    Filter2DCache::Ref filter2d_cache_;

    mutable std::mutex mutex_;
};

} // namespace holovibes
