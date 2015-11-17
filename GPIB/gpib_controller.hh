#pragma once

# include <vector>
# include <deque>
# include <exception>
# include <utility>
# include <iostream>

# define BUF_SIZE 200

namespace gpib
{
  /*! Contains all elements needed to establish
   * a connection to a device through the VISA interface. */
  class VisaInterface
  {
  public:
    /*! Build an interface to the GPIB driver using the VISA standard.
     * \param path Path to the batch file that is provided to control
     * images recording using GPIB-driven components. */
    VisaInterface(const std::string& path);

    /*! Making sure all opened connections are closed,
     * and freeing allocated memory. */
    ~VisaInterface();

    VisaInterface(const VisaInterface& other) = delete;

    VisaInterface& operator=(const VisaInterface& other) = delete;

    /*! Setting up the connection with an instrument at a given address. */
    void initialize_instr(const unsigned address);

    /*! Closing the connection with a given instrument, knowing its address. */
    void close_instr(const unsigned address);

    /*! Launch the commands extracted previously from the input file.
     * \return True if there are more commands to issue. */
    bool execute_next_block();

  private:
    /*! Setting up the VISA driver to enable future connections.
     * Automatically called by the constructor. */
    void initialize_line();

    /*! Closing the connection to the VISA driver.
     * Automatically called by the destructor. */
    void close_line();

    /*! Parse the file and report any error in the format. */
    void parse_file(std::ifstream& in);

  private:
    /*! To decouple dependencies between the GPIB controller and Holovibes,
     * a kind of Pimpl idiom is used. In this case, we do not use an intermediate
     * Pimpl class, but just regroup all VISA-related types in a structure,
     * and include visa.h in the implementation file (gpib_controller.cc).
     * Hence Holovibes is not dependent of visa.h by including gpib_controller.hh.
     */
    struct VisaPimpl;
    VisaPimpl* pimpl_;

    /*! Each command is formed of an instrument address,
    * a proper command sent as a string through the VISA interface,
    * and a number of milliseconds to wait for until next command
    * is issued. */
    struct Command
    {
      unsigned address;
      std::string command;
      unsigned wait;
    };

    /*! Lines obtained from the batch input file are stored
     * here as separate strings. */
    std::deque<Command> batch_cmds_;
  };
}