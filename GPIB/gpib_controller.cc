#include <iostream>
#include <string>
#include <algorithm>
#include <boost/lexical_cast.hpp>

#include "visa.h"

#include "gpib_controller.hh"
#include "gpib_exceptions.hh"

namespace gpib
{
/*! \brief A connection is composed of a VISA session and a device address. */
using instrument = std::pair<ViSession, unsigned>;

struct VisaInterface::VisaPimpl
{
    VisaPimpl()
        : status_{VI_SUCCESS}
        , ret_count_{0}
        , buffer_{nullptr}
    {
    }

    /*! \brief Error status */
    ViStatus status_;

    /*! \brief Session used to open/close the VISA driver. */
    ViSession default_rm_;
    /*! \brief Active connections. */
    std::vector<instrument> sessions_;

    /*! \brief Counting the number of characters returned by a read. */
    ViPUInt32 ret_count_;
    /*! \brief Buffer used for writing/reading. */
    ViPByte buffer_;
};

VisaInterface::VisaInterface()
    : pimpl_{new VisaPimpl()}
{
}

VisaInterface::~VisaInterface()
{
    if (pimpl_->buffer_)
        delete[] pimpl_->buffer_;

    std::for_each(pimpl_->sessions_.begin(),
                  pimpl_->sessions_.end(),
                  [this](instrument& instr) { close_instrument(instr.second); });

    close_line();

    delete pimpl_;
}

void VisaInterface::initialize_instrument(const unsigned address)
{
    // Timeout value used when waiting from a GPIB device.
    const unsigned timeout = 5000;
    std::string address_msg("GPIB0::");
    address_msg.append(boost::lexical_cast<std::string>(address));
    address_msg.append("::INSTR");

    pimpl_->sessions_.push_back(instrument(0, address));
    pimpl_->status_ = viOpen(pimpl_->default_rm_,
                             (ViString)(address_msg.c_str()),
                             VI_NULL,
                             VI_NULL,
                             &(pimpl_->sessions_.back().first));
    if (pimpl_->status_ != VI_SUCCESS)
    {
        // std::cerr << "[GPIB] Could not set up connection with instrument " << address << std::endl;
        throw GpibInstrError(boost::lexical_cast<std::string>(address));
    }

    viSetAttribute(pimpl_->sessions_.back().first, VI_ATTR_TMO_VALUE, timeout);
}

void VisaInterface::close_instrument(const unsigned address)
{
    auto it = std::find_if(pimpl_->sessions_.begin(),
                           pimpl_->sessions_.end(),
                           [address](instrument& instr) { return instr.second == address; });

    if (it != pimpl_->sessions_.end())
    {
        // Closing the session, and removing the session from the sessions
        // vector.
        pimpl_->status_ = viClose(it->first);
        if (pimpl_->status_ == VI_SUCCESS)
            pimpl_->sessions_.erase(it);
    }
}

void VisaInterface::execute_instrument_command(const BatchCommand& instrument_command)
{
    assert(instrument_command.type == BatchCommand::INSTRUMENT_COMMAND);

    /* If this is the first time a command is issued, the connexion
     * with the VISA interface must be set up.
     */
    if (!pimpl_->buffer_)
    {
        try
        {
            initialize_line();
        }
        catch (const std::exception& /*e*/)
        {
            throw;
        }
    }
    /* If a connexion to this instrument address is not opened,
     * do it and register the new session.
     */
    if (std::find_if(pimpl_->sessions_.begin(),
                     pimpl_->sessions_.end(),
                     [&instrument_command](instrument& instr)
                     { return instr.second == instrument_command.address; }) == pimpl_->sessions_.end())
    {
        initialize_instrument(instrument_command.address);
    }

    // Get the session and send it the command through VISA.
    auto ses =
        std::find_if(pimpl_->sessions_.begin(),
                     pimpl_->sessions_.end(),
                     [&instrument_command](instrument& instr) { return instr.second == instrument_command.address; });
    viWrite(ses->first,
            (ViBuf)(instrument_command.command.c_str()), // ViBuf it's so crap type,
                                                         // that no c++ cast works
            static_cast<ViInt32>(instrument_command.command.size()),
            pimpl_->ret_count_);
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
    /* VisaPimpl's buffer's allocation assures Visa has been used,
     * and so that a connection was set up.
     */
    if (pimpl_->buffer_ && viClose(pimpl_->default_rm_) != VI_SUCCESS)
    {
        // std::cerr << "[GPIB] Could not close connection to VISA driver." << std::endl;
    }
}

IVisaInterface* new_gpib_controller() { return new VisaInterface(); }
} // namespace gpib
