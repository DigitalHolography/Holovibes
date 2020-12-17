/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "information_container.hh"

namespace holovibes
{
inline void InformationContainer::set_update_progress_function(
    const std::function<
        void(InformationContainer::ProgressType, size_t, size_t)>& function)
{
    update_progress_function_ = function;
}

inline void InformationContainer::set_display_info_text_function(
    const std::function<void(const std::string&)>& function)
{
    display_info_text_function_ = function;
}
} // namespace holovibes