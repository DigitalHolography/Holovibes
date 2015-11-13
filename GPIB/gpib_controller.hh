#pragma once

# include <vector>
# include <deque>
# include <exception>
# include <utility>
# include <iostream>

# define BUF_SIZE 200

/*! Contains all elements needed to establish
 * a connection to a device through the VISA interface. */
class VisaInterface
{
public:
  VisaInterface(const std::string& path);

  ~VisaInterface();

  /*! Setting up the connection with an instrument at a given address. */
  void initialize_instr(const unsigned address);

  /*! Closing the connection with a given instrument, knowing its address. */
  void close_instr(const unsigned address);

  /*! Launch the commands extracted previously from the input file. */
  int execute_next_block();

private:
  /*! Setting up the VISA driver to enable future connections.
   * Automatically called by the constructor. */
  void initialize_line();

  /*! Closing the connection to the VISA driver.
   * Automatically called by the destructor. */
  void close_line();

private:
  struct VisaPimpl;
  VisaPimpl* pimpl_;

  /*! Lines obtained from the batch input file are stored
   * here as separate strings. */
  std::deque<std::string> batch_cmds_;
};