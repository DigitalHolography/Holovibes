#pragma once

# include "visa.h"
# include <vector>
# include <exception>
# include <utility>
# include <iostream>

# define BUF_SIZE 200

//!< A connection is composed of a VISA session and a device address.
using instrument = std::pair<ViSession, unsigned>;

/*! Contains all elements needed to establish
 * a connection to a device through the VISA interface. */
class VisaInterface
{
public:
  VisaInterface();

  ~VisaInterface();

  /*! Setting up the connection with an instrument at a given address. */
  void initialize_instr(const unsigned address);

  /*! Closing the connection with a given instrument, knowing its address. */
  void close_instr(const unsigned address);

private:
  /*! Setting up the VISA driver to enable future connections.
   * Automatically called by the constructor. */
  void initialize_line();

  /*! Closing the connection to the VISA driver.
   * Automatically called by the destructor. */
  void close_line();

private:
  ViStatus status_; //!< Error status

  ViSession default_rm_; //!< Session used to open/close the VISA driver.
  std::vector<std::pair<ViSession, unsigned>> sessions_; //!< Active connections.

  ViPUInt32 ret_count_; //!< Counting the number of characters returned by a read.
  ViPByte buffer_; //!< Buffer used for writing/reading.
};