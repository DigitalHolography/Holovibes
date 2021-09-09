#pragma once

#include <vector>
#include <exception>
#include <utility>

#include "IVisaInterface.hh"

#define BUF_SIZE 200

namespace gpib
{
/*! Contains all elements needed to establish
 * a connection to a device through the VISA interface. */
class VisaInterface : public IVisaInterface
{
  public:
    /*! Build an interface to the GPIB driver using the VISA standard. */
    VisaInterface();

    /*! Making sure all opened connections are closed,
     * and freeing allocated memory. */
    virtual ~VisaInterface();

    VisaInterface(const VisaInterface& other) = delete;

    VisaInterface& operator=(const VisaInterface& other) = delete;

    void execute_instrument_command(const BatchCommand& cmd);

  private:
    /*! Setting up the connection with an instrument at a given address. */
    void initialize_instrument(const unsigned address);

    /*! Closing the connection with a given instrument, knowing its address. */
    void close_instrument(const unsigned address);

    /*! Setting up the VISA driver to enable future connections. */
    void initialize_line();

    /*! Closing the connection to the VISA driver.
     * Automatically called by the destructor. */
    void close_line();

  private:
    /*! To decouple dependencies between the GPIB controller and Holovibes,
     * a kind of Pimpl idiom is used. In this case, we do not use an
     * intermediate Pimpl class, but just regroup all VISA-related types in a
     * structure, and include visa.h in the implementation file
     * (gpib_controller.cc). Hence Holovibes is not dependent of visa.h by
     * including gpib_controller.hh.
     */
    struct VisaPimpl;
    VisaPimpl* pimpl_;
};
} // namespace gpib