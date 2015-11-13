#include <algorithm>

#include "gpib_controller.hh"
#include "include\gpib.h"

int load_batch_file(const char* filepath)
{
  return 666;
}

int execute_next_block(void)
{
  return 666;
}

VisaInterface::VisaInterface()
: status_ { VI_SUCCESS }
, ret_count_ { 0 }
, buffer_ { nullptr }
{
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
}

VisaInterface::~VisaInterface()
{
  free(buffer_);

  std::for_each(sessions_.begin(),
    sessions_.end(),
    [this](instrument& instr)
  {
    close_instr(instr.first);
  });
  sessions_.clear();

  close_line();
}

void VisaInterface::initialize_instr(const unsigned address)
{
  // Timeout value used when waiting from a GPIB device.
  const unsigned timeout = 5000;

  sessions_.push_back(instrument(0, address));
  status_ = viOpen(default_rm_,
    "GPIB0::20::INSTR", // TODO : Convert address to string
    VI_NULL,
    VI_NULL,
    &sessions_.back().first);
  if (status_ != VI_SUCCESS)
  {
    std::cerr << "[GPIB] Could not set up connection with instrument " << address << "\n";
    throw std::bad_exception();
  }

  viSetAttribute(sessions_.back().first, VI_ATTR_TMO_VALUE, timeout);
}

void VisaInterface::close_instr(const unsigned address)
{
  auto it = std::find_if(sessions_.begin(),
    sessions_.end(),
    [address](instrument& instr)
  {
    return instr.second == address;
  });

  if (it != sessions_.end())
  {
    status_ = viClose(it->first);
    sessions_.erase(it);
  }
}

void VisaInterface::initialize_line()
{
  buffer_ = static_cast<ViPByte>(malloc(sizeof(ViByte)* BUF_SIZE));

  if (!buffer_)
    throw std::bad_alloc();

  // Setting up connection with the VISA driver.
  status_ = viOpenDefaultRM(&default_rm_);
  if (status_ != VI_SUCCESS)
    throw std::bad_exception();
}

void VisaInterface::close_line()
{
  if (viClose(default_rm_))
    std::cerr << "[GPIB] Could not close connection to VISA driver.\n";
}