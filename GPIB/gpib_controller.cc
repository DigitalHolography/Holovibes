/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include <string>
#include <fstream>
#include <algorithm>
#include <thread>
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
		VisaPimpl() : status_{ VI_SUCCESS }
			, ret_count_{ 0 }
			, buffer_{ nullptr }
		{}

		ViStatus status_; //!< Error status

		ViSession default_rm_; //!< Session used to open/close the VISA driver.
		std::vector<instrument> sessions_; //!< Active connections.

		ViPUInt32 ret_count_; //!< Counting the number of characters returned by a read.
		ViPByte buffer_; //!< Buffer used for writing/reading.
	};

	VisaInterface::VisaInterface(const std::string& path)
		: pimpl_{ new VisaPimpl() },
		path_(path)
	{
		std::ifstream in;
		in.open(path);
		if (!in.is_open())
			throw std::exception("GPIB : Invalid Filepath");

		try
		{
			parse_file(in);
		}
		catch (const std::exception&)
		{
			throw;
		}

		in.close();
	}

	VisaInterface::~VisaInterface()
	{
		if (pimpl_->buffer_)
			delete[] pimpl_->buffer_;

		std::for_each(pimpl_->sessions_.begin(),
			pimpl_->sessions_.end(),
			[this](instrument& instr)
		{
			close_instrument(instr.second);
		});

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
			std::cerr << "[GPIB] Could not set up connection with instrument " << address << std::endl;
			throw GpibInstrError(boost::lexical_cast<std::string>(address));
		}

		viSetAttribute(pimpl_->sessions_.back().first, VI_ATTR_TMO_VALUE, timeout);
	}

	void VisaInterface::close_instrument(const unsigned address)
	{
		auto it = std::find_if(pimpl_->sessions_.begin(),
			pimpl_->sessions_.end(),
			[address](instrument& instr)
		{
			return instr.second == address;
		});

		if (it != pimpl_->sessions_.end())
		{
			// Closing the session, and removing the session from the sessions vector.
			pimpl_->status_ = viClose(it->first);
			if (pimpl_->status_ == VI_SUCCESS)
				pimpl_->sessions_.erase(it);
		}
	}

	std::optional<Command> VisaInterface::get_next_command()
	{
		if (batch_cmds_.empty())
			return std::nullopt;
		
		return batch_cmds_.back();
	}

	void VisaInterface::pop_next_command()
	{
		assert(!batch_cmds_.empty());

		batch_cmds_.pop_back();
	}

	void VisaInterface::execute_instrument_command(const Command& instrument_command)
	{
		assert(instrument_command.type == Command::INSTRUMENT_COMMAND);

		/* If this is the first time a command is issued, the connexion
			* with the VISA interface must be set up. */
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
		* do it and register the new session. */
		if (std::find_if(pimpl_->sessions_.begin(),
			pimpl_->sessions_.end(),
			[&instrument_command](instrument& instr)
		{
			return instr.second == instrument_command.address;
		}) == pimpl_->sessions_.end())
		{
			initialize_instrument(instrument_command.address);
		}

		// Get the session and send it the command through VISA.
		auto ses = std::find_if(pimpl_->sessions_.begin(),
			pimpl_->sessions_.end(),
			[&instrument_command](instrument& instr)
		{
			return instr.second == instrument_command.address;
		});
		viWrite(ses->first,
			(ViBuf)(instrument_command.command.c_str()), //ViBuf it's so crap type, that no c++ cast works
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
		 * and so that a connection was set up. */
		if (pimpl_->buffer_ && viClose(pimpl_->default_rm_) != VI_SUCCESS)
			std::cerr << "[GPIB] Could not close connection to VISA driver." << std::endl;
	}

	void VisaInterface::parse_file(std::ifstream& in)
	{
		std::string line;
		unsigned int line_num = 0;
		unsigned int cur_address = 0;

		while (in >> line)
		{
			batch_cmds_.push_front(Command());
			Command& cmd = batch_cmds_.front();

			if (line.compare("#Block") == 0)
			{
				// Just a #Block : no special meaning.
				batch_cmds_.pop_front();
			}
			else if (line.compare("#InstrumentAddress") == 0)
			{
				// We change the address currently used for commands.
				try
				{
					in >> line;
					unsigned int address = boost::lexical_cast<unsigned>(line);
					cur_address = address;
					batch_cmds_.pop_front();
				}
				catch (const boost::bad_lexical_cast& /*e*/)
				{
					throw GpibParseError(boost::lexical_cast<std::string>(line_num),
						GpibParseError::NoAddress);
				}
			}
			else if (line.compare("#WAIT") == 0)
			{
				// We insert a waiting action in the block.
				try
				{
					in >> line;
					unsigned int wait = boost::lexical_cast<unsigned>(line);

					cmd.type = Command::WAIT;
					cmd.address = 0;
					cmd.command = "";
					cmd.wait = wait;
				}
				catch (const boost::bad_lexical_cast& /*e*/)
				{
					throw GpibParseError(boost::lexical_cast<std::string>(line_num),
						GpibParseError::NoWait);
				}
			}
			else if (line.compare("#Capture") == 0)
			{
				cmd.type = Command::CAPTURE;
				cmd.address = 0;
				cmd.command = "";
				cmd.wait = 0;
			}
			else
			{
				/* A command string, the validity of which can not be tested because of
				 * the multiple interfaces to various existing instruments. */
				for (unsigned i = 0; i < line.size(); ++i)
					in.unget();
				std::getline(in, line, '\n');
				line.append("\n"); // Don't forget the end-of-command character for VISA.

				cmd.type = Command::INSTRUMENT_COMMAND;
				cmd.address = cur_address;
				cmd.command = line;
				cmd.wait = 0;
			}
			++line_num;
		}

		if (line_num == 0)
		{
			// We just read a blank file...
			throw GpibBlankFileError();
		}
	}

	IVisaInterface* new_gpib_controller(const std::string path)
	{
		return new VisaInterface(path);
	}
}
