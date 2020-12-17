/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Implementation of the Observable design pattern. */
#pragma once

#include <algorithm>
#include <vector>
#include "observer.hh"

namespace holovibes
{
/*! \brief Implementation of the Observable design pattern.
 *
 * Set a child class to be Observable and contains a list of Observer that
 * will be notified when the Observable class is modified. */
class Observable
{
  public:
    /*! \brief add Observer in list*/
    void register_observer(Observer& o) { observers_.push_back(&o); }

    /*! \brief notify all Observer in list */
    void notify_observers()
    {
        std::for_each(observers_.begin(),
                      observers_.end(),
                      [](Observer* observer) { observer->notify(); });
    }

    /*! \brief notify all Observer in list that an error occured */
    void notify_error_observers(std::exception& e)
    {
        std::for_each(observers_.begin(),
                      observers_.end(),
                      [&](Observer* observer) { observer->notify_error(e); });
    }

  protected:
    Observable() {}

    virtual ~Observable() {}

  private:
    std::vector<Observer*> observers_;
};
} // namespace holovibes