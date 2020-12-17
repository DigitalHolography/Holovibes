/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
        BLOCK,   // #Block : ignored, just for clarity
        CAPTURE, // #Capture : Stop issuing commands and acquire a frame
        INSTRUMENT_COMMAND, // * : Sent to an instrument as is in a message
                            // buffer
        WAIT                // #WAIT n : Put the thread to sleep n milliseconds
    };

    type_e type;

    unsigned address;
    std::string command;
    unsigned wait;
};
} // namespace gpib