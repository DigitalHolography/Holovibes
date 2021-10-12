#pragma once

#include <unordered_map>
#include <memory>
#include <stdexcept>

#include "fast_updates_types.hh"

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
    using Value = std::shared_ptr<FastUpdateTypeValue<T>>;

    /*!
     * \brief Create a fast update entry object in the map of type T
     *
     * \param key The key of an enum T from the fast_updates_types.hh
     * \param overwrite it there a need to overwrite the previous entry ?
     * \return std::shared_ptr<Value> The pointer returned to the entry in the map
     */
    std::shared_ptr<Value> create_entry(Key key, bool overwrite = false)
    {
        if (map_.contains(key) && !overwrite)
            throw std::runtime_error("Key is already present in map");

        return map_[key] = std::make_shared<Value>;
    }

    /*!
     * \brief Get the entry object
     *
     * \param key The key of an enum T from the fast_updates_types.hh
     * \return std::shared_ptr<Value> The pointer returned to the entry in the map
     */
    std::shared_ptr<Value> get_entry(Key key) const { return map_.at(key); }

    /*!
     * \brief Remove an entry from the map
     *
     * \param key The key of an enum T from the fast_updates_types.hh
     * \return true The entry was deleted
     * \return false The entry was not deleted
     */
    bool remove_entry(Key key)
    {
        auto it = map_.find(key);
        if (it == map_.end())
            return false;

        map_.erase(it);
        return true;
    }

    /*! \brief Clears the map */
    void clear() { map_.clear(); }

  protected:
    const std::unordered_map<Key, Value> map_;
};

} // namespace holovibes