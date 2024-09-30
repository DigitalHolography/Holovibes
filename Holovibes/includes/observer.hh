/*! \file
 *
 * \brief Implementation of the Observer design pattern.
 */
#pragma once

namespace holovibes
{
/*! \class Observer
 *
 * \brief Implementation of the Observer design pattern.
 *
 * Set a child class to be observer of an observable object.
 */
class Observer
{
  public:
    /*! \brief Notify method called when Observable class change.
     *
     * Mandatory method that is called when an Observable object has changed of state.
     */
    virtual void notify() = 0;

    virtual void notify_error(const std::exception& e) = 0;

  protected:
    Observer() {}

    virtual ~Observer() {}
};
} // namespace holovibes
