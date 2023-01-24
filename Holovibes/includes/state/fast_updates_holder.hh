#pragma once

#include <unordered_map>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "fast_updates_types.hh"
#include "logger.hh"

namespace holovibes
{
template <class T>
class FastUpdatesHolder
{
    // Check if the template parameter is valid
    // Cf. fast_updates_types.hh
    static_assert(is_fast_update_key_type<T>);

  public:
    using Key = T;
    using Value = FastUpdateTypeValue<T>;
    using const_iterator = typename std::unordered_map<Key, Value>::const_iterator;
    using iterator = typename std::unordered_map<Key, Value>::iterator;

    /*!
     * \brief Create a fast update entry object in the map of type T
     *
     * \param key The key of an enum T from the fast_updates_types.hh
     * \param overwrite it there a need to overwrite the previous entry ?
     * \return std::shared_ptr<Value> The pointer returned to the entry in the map
     */
    Value& create_entry(Key key, bool overwrite = false)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!overwrite && map_.contains(key))
        {
            LOG_WARN("Key is already defined in the map {}", typeid(T).name());
        }

        map_[key] = FastUpdateTypeValue<T>();

#ifndef DISABLE_LOG_UPDATE_MAP_ENTRY
        LOG_DEBUG("New FastUpdatesHolder<{}> {}", typeid(T).name(), key);
#endif

        return map_[key];
    }

    /*!
     * \brief Get the entry object
     *
     * \param key The key of an enum T from the fast_updates_types.hh
     * \return std::shared_ptr<Value> The pointer returned to the entry in the map
     */
    Value& get_entry(Key key) { return map_.at(key); }

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

    /*! \brief Iterators */

    const_iterator begin() { return map_.begin(); }
    const_iterator end() { return map_.end(); }

    const_iterator find(Key key) const { return map_.find(key); }

    std::unordered_map<Key, Value>& get_map() { return map_; }

  protected:
    std::mutex mutex_;
    std::unordered_map<Key, Value> map_;
};
} // namespace holovibes
