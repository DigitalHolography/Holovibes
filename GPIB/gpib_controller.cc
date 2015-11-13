#include <string>
#include <fstream>
#include <algorithm>
# include "visa.h"

#include "gpib_controller.hh"

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
  std::vector<std::pair<ViSession, unsigned>> sessions_; //!< Active connections.

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
  catch (const std::bad_alloc& e)
  {
    std::cerr << "[GPIB] Could not allocate buffer.\n";
  }
  catch (const std::bad_exception& e)
  {
    std::cerr << "[GPIB] Could not set up VISA communication.\n";
  }

  // Batch input file parsing
  std::ifstream in;
  in.open(path);

  if (!in.is_open())
  {
    delete[] pimpl_->buffer_;
    throw std::bad_exception();
  }

  std::string line;
  while (in >> line)
    batch_cmds_.push_front(line);
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
  pimpl_->sessions_.clear();

  close_line();

  delete pimpl_;
}

void VisaInterface::initialize_instr(const unsigned address)
{
  // Timeout value used when waiting from a GPIB device.
  const unsigned timeout = 5000;

  pimpl_->sessions_.push_back(instrument(0, address));
  pimpl_->status_ = viOpen(pimpl_->default_rm_,
    "GPIB0::20::INSTR", // TODO : Convert address to string
    VI_NULL,
    VI_NULL,
    &(pimpl_->sessions_.back().first));
  if (pimpl_->status_ != VI_SUCCESS)
  {
    std::cerr << "[GPIB] Could not set up connection with instrument " << address << "\n";
    throw std::bad_exception();
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
  return 0;
}

void VisaInterface::initialize_line()
{
  pimpl_->buffer_ = new ViByte[BUF_SIZE];

  if (!pimpl_->buffer_)
    throw std::bad_alloc();

  // Setting up connection with the VISA driver.
  pimpl_->status_ = viOpenDefaultRM(&pimpl_->default_rm_);
  if (pimpl_->status_ != VI_SUCCESS)
    throw std::bad_exception();
}

void VisaInterface::close_line()
{
  if (viClose(pimpl_->default_rm_))
    std::cerr << "[GPIB] Could not close connection to VISA driver.\n";
}