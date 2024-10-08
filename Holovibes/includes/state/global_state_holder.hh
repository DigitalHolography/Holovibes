/*! \file
 *
 * \brief Global State Holder Class
 *
 *  Holds the state of the entire application
 */

#pragma once

#include <mutex>

#include "fast_updates_holder.hh"

namespace holovibes
{

/*! \class GSH
 *
 * \brief The GSH (global state holder), is where all the global state of the program is stored.
 *
 * FastUpdateHolder : the fastUpdateHolder is a templated map which is used by the informationWorker to access and
 * display information (like fps and queue occupancy) at a high rate, since this needs to be updated continuously.
 */
class GSH
{
  public:
    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

    enum class ComputeSettingsVersion
    {
        V2,
        V3,
        V4,
        V5
    };
    static void convert_json(json& data, GSH::ComputeSettingsVersion from);

  private:
    GSH() noexcept {}

};

} // namespace holovibes
