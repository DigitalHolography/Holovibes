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

#pragma once

# include "IVisaInterface.hh"


# define BUF_SIZE 200

namespace gpib
{
	/*! Contains all elements needed to establish
	 * a connection to a device through the VISA interface. */
	class VisaInterface : public IVisaInterface
	{
	public:
		/*! Build an interface to the GPIB driver using the VISA standard.
		 * \param path Path to the batch file that is provided to control
		 * images recording using GPIB-driven components. */
		VisaInterface(const std::string& path);

		/*! Making sure all opened connections are closed,
		 * and freeing allocated memory. */
		virtual ~VisaInterface();

		VisaInterface(const VisaInterface& other) = delete;

		VisaInterface& operator=(const VisaInterface& other) = delete;

		/*! Setting up the connection with an instrument at a given address. */
		void initialize_instr(const unsigned address);

		/*! Closing the connection with a given instrument, knowing its address. */
		void close_instr(const unsigned address);

		/*! Launch the commands extracted previously from the input file.
		 * \return True if there are more commands to issue. */
		virtual bool	execute_next_block();
		bool			execute_next_trig();


		std::shared_ptr<gpib::IVisaInterface> gpib_interface_;
		std::shared_ptr<gpib::IVisaInterface> get_gpib_interface() { return gpib_interface_; }

	private:
		/*! Setting up the VISA driver to enable future connections. */
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
			enum type_e
			{
				BLOCK,   // #Block   : ignored, just for clarity
				CAPTURE, // #Capture : Stop issuing commands and acquire a frame
				COMMAND, // *        : Sent to an instrument as is in a message buffer
				WAIT     // #WAIT n  : Put the thread to sleep n milliseconds
			};

			type_e type;

			unsigned address;
			std::string command;
			unsigned wait;
		};

		/*! Lines obtained from the batch input file are stored
		 * here as separate strings. */
		std::deque<Command> batch_cmds_;
		std::string			path_;

	};
}