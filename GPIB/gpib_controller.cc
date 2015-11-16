#include <string>
#include <fstream>
#include <algorithm>
#include <boost/lexical_cast.hpp>
#include "visa.h"

#include "gpib_controller.hh"
#include "gpib_exceptions.hh"

namespace gpib
{
  //!< A connection is composed of a VISA session and a device address.
  using instrument = std::pair<ViSession, unsigned>;

  struct VisaInterface::VisaPimpl
  {
    VisaPimpl()
    : status_ { VI_SUCCESS }
    , ret_count_ { 0 }
    , buffer_ { nullptr }
    {
    }

    ViStatus status_; //!< Error status

    ViSession default_rm_; //!< Session used to open/close the VISA driver.
    std::vector<instrument> sessions_; //!< Active connections.

    ViPUInt32 ret_count_; //!< Counting the number of characters returned by a read.
    ViPByte buffer_; //!< Buffer used for writing/reading.
  };

  VisaInterface::VisaInterface(const std::string& path)
    : pimpl_ { new VisaPimpl() }
  {
    // Basic initializations
    try
    {
      initialize_line();
    }
    catch (const GpibBadAlloc& e)
    {
      throw;
    }
    catch (const GpibSetupError& e)
    {
      throw;
    }

    // Batch input file parsing
    if (path.compare("==") == 0)
      throw GpibNoFilepath();

    std::ifstream in;
    in.open(path);
    if (!in.is_open())
    {
      delete[] pimpl_->buffer_;
      throw GpibInvalidPath(path);
    }

    try
    {
      parse_file(in);
    }
    catch (const GpibParseError& e)
    {
      throw;
    }

    in.close();
  }

  VisaInterface::~VisaInterface()
  {
    delete[] pimpl_->buffer_;

    std::for_each(pimpl_->sessions_.begin(),
      pimpl_->sessions_.end(),
      [this](instrument& instr)
    {
      close_instr(instr.first);
    });

    close_line();

    delete pimpl_;
  }

  void VisaInterface::initialize_instr(const unsigned address)
  {
    // Timeout value used when waiting from a GPIB device.
    const unsigned timeout = 5000;
    std::string address_msg("GPIB0::");
    address_msg.append(boost::lexical_cast<std::string>(address));
    address_msg.append("::INSTR");

    pimpl_->sessions_.push_back(instrument(0, address));
    pimpl_->status_ = viOpen(pimpl_->default_rm_,
      (ViString)(address_msg.c_str()), // TODO : Try to replace with a C++-cast
      VI_NULL,
      VI_NULL,
      &(pimpl_->sessions_.back().first));
    if (pimpl_->status_ != VI_SUCCESS)
    {
      std::cerr << "[GPIB] Could not set up connection with instrument " << address << "\n";
      throw GpibInstrError(boost::lexical_cast<std::string>(address));
    }

    viSetAttribute(pimpl_->sessions_.back().first, VI_ATTR_TMO_VALUE, timeout);
  }

  void VisaInterface::close_instr(const unsigned address)
  {
    auto it = std::find_if(pimpl_->sessions_.begin(),
      pimpl_->sessions_.end(),
      [address](instrument& instr)
    {
      return instr.second == address;
    });

    if (it != pimpl_->sessions_.end())
    {
      pimpl_->status_ = viClose(it->first);
      pimpl_->sessions_.erase(it);
    }
  }

  int VisaInterface::execute_next_block()
  {
    auto cmd = pimpl_->sessions_.back();

    return 0;
  }

  void VisaInterface::initialize_line()
  {
    pimpl_->buffer_ = new ViByte[BUF_SIZE];
    if (!pimpl_->buffer_)
      throw GpibBadAlloc();

    // Setting up connection with the VISA driver.
    pimpl_->status_ = viOpenDefaultRM(&pimpl_->default_rm_);
    if (pimpl_->status_ != VI_SUCCESS)
      throw GpibSetupError();
  }

  void VisaInterface::close_line()
  {
    if (viClose(pimpl_->default_rm_))
      std::cerr << "[GPIB] Could not close connection to VISA driver.\n";
  }

  void VisaInterface::parse_file(std::ifstream& in)
  {
    std::string line;
    unsigned line_num = 0;

    while (in >> line)
    {
      batch_cmds_.push_front(Command());
      Command& cmd = batch_cmds_.front();

      // Just the preamble #Block.
      if (line.compare("#Block") != 0)
        throw GpibParseError(boost::lexical_cast<std::string>(line_num),
        GpibParseError::NoBlock);
      ++line_num;

      // Parsing the instrument address.
      in >> line;
      if (line.substr(0, 18).compare("#InstrumentAddress") != 0)
        throw GpibParseError(boost::lexical_cast<std::string>(line_num),
        GpibParseError::NoAddress);
      try
      {
        unsigned address = boost::lexical_cast<unsigned>(line.substr(19, line.size()));
        cmd.address = address;
      }
      catch (const boost::bad_lexical_cast& e)
      {
        throw GpibParseError(boost::lexical_cast<std::string>(line_num),
          GpibParseError::NoAddress);
      }
      ++line_num;

      // Getting the raw string to be sent to the instrument.
      in >> line;
      cmd.command = line;
      ++line_num;

      in >> line;
      if (line.substr(0, 5).compare("#WAIT") != 0)
        throw GpibParseError(boost::lexical_cast<std::string>(line_num),
        GpibParseError::NoWait);
      try
      {
        unsigned wait = boost::lexical_cast<unsigned>(line.substr(6, line.size()));
        cmd.wait = wait;
      }
      catch (const boost::bad_lexical_cast& e)
      {
        throw GpibParseError(boost::lexical_cast<std::string>(line_num),
          GpibParseError::NoWait);
      }
      ++line_num;
    }
  }
}