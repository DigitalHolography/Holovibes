/*! \file fast_updates_holder.hh
 *
 * \brief Contains the definition of the FastUpdatesHolder class, used to store and access information at a high rate.
 */
#pragma once

#include <unordered_map>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "fast_updates_types.hh"
#include "logger.hh"

namespace holovibes
{

/*! \class FastUpdatesHolder
 *
 * \brief Hold a templated map which is used by the informationWorker to access and
 * display information (like fps and queue occupancy) at a high rate, since this needs to be updated continuously.
 */
template <class T>
class FastUpdatesHolder
{
    // Check if the template parameter is valid
    // Cf. fast_updates_types.hh
    static_assert(is_fast_update_key_type<T>);

  public:
    using Key = T;
    using Value = std::shared_ptr<FastUpdateTypeValue<T>>;
    using const_iterator = typename std::unordered_map<Key, Value>::const_iterator;

    /*!
     * \brief Create a fast update entry object in the map of type T
     *
     * \param key The key of an enum T from the fast_updates_types.hh
     * \param overwrite it there a need to overwrite the previous entry ?
     * \return std::shared_ptr<Value> The pointer returned to the entry in the map
     */
    Value create_entry(Key key, bool overwrite = false)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!overwrite && map_.contains(key))
            throw std::runtime_error("Key is already present in map");

        map_[key] = std::make_shared<FastUpdateTypeValue<T>>();

        LOG_DEBUG("New FastUpdatesHolder<{}> entry: 0x{}", typeid(T).name(), map_[key]);

        return map_[key];
    }

    /*!
     * \brief Get the entry object, create it if it does not exist
     *
     * \param[in] key The key of an enum T from the fast_updates_types.hh
     * \return std::shared_ptr<Value> The pointer returned to the entry in the map
     */
    Value get_or_create_entry(Key key)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!map_.contains(key))
            map_[key] = std::make_shared<FastUpdateTypeValue<T>>();

        LOG_DEBUG("New FastUpdatesHolder<{}> entry: 0x{}", typeid(T).name(), map_[key]);

        return map_[key];
    }

    /*!
     * \brief Get the entry object
     *
     * \param key The key of an enum T from the fast_updates_types.hh
     * \return std::shared_ptr<Value> The pointer returned to the entry in the map
     */
    Value get_entry(Key key) const { return map_.at(key); }

    /*!
     * \brief Remove an entry from the map
     *
     * \param key The key of an enum T from the fast_updates_types.hh
     * \return true The entry was deleted
     * \return false The entry was not deleted
     */
    bool remove_entry(Key key)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(key);
        if (it == map_.end())
            return false;

        map_.erase(it);
        return true;
    }

    bool contains(Key key) const noexcept { return map_.contains(key); }

    /*! \brief Clears the map */
    void clear() noexcept
    {
        std::lock_guard<std::mutex> lock(mutex_);
        map_.clear();
    }

    bool empty() const noexcept { return map_.empty(); }

    /*! \brief Iterators */

    const_iterator begin() { return map_.begin(); }
    const_iterator end() { return map_.end(); }

    const_iterator find(Key key) { return map_.find(key); }

  protected:
    std::mutex mutex_;
    std::unordered_map<Key, Value> map_;
};

/*! \class FastUpdatesMap
 *
 * \brief Container of all the FastUpdatesHolder map used by the application.
 */
struct FastUpdatesMap
{
    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> map;
};

} // namespace holovibes
