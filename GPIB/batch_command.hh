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

namespace gpib
{
    /*! Each command is formed of an instrument address,
    * a proper command sent as a string through the VISA interface,
    * and a number of milliseconds to wait for until next command
    * is issued. */
    struct BatchCommand
    {
        enum type_e
        {
            BLOCK,   			// #Block : ignored, just for clarity
            CAPTURE, 			// #Capture : Stop issuing commands and acquire a frame
            INSTRUMENT_COMMAND, // * : Sent to an instrument as is in a message buffer
            WAIT     			// #WAIT n : Put the thread to sleep n milliseconds
        };

        type_e type;

        unsigned address;
        std::string command;
        unsigned wait;
    };
} // namespace gpib