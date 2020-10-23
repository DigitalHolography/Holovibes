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

#include "ithread_input.hh"

#include "logger.hh"

namespace holovibes
{
  IThreadInput::IThreadInput()
    : stop_requested_(false)
  {
  }

  IThreadInput::~IThreadInput()
  {
  }

  SquareInputMode get_square_input_mode_from_string(const std::string &name)
  {
    if (name == "Zero Padded")
    {
      return SquareInputMode::ZERO_PADDED_SQUARE;
    }
    else if (name == "Cropped")
    {
      return SquareInputMode::CROPPED_SQUARE;
    }
    else if (name == "Default")
    {
      return SquareInputMode::NO_MODIFICATION;
    }
    else
    {
      LOG_WARN(std::string("Unsupported square input mode : ") + name);
      return SquareInputMode::NO_MODIFICATION;
    }
  }

  std::string get_string(const SquareInputMode mode)
  {
    switch (mode)
    {
      case SquareInputMode::NO_MODIFICATION:
        return std::string("NO_MODIFICATION");
      case SquareInputMode::ZERO_PADDED_SQUARE:
        return std::string("ZERO_PADDED_SQUARE");
      case SquareInputMode::CROPPED_SQUARE:
        return std::string("CROPPED_SQUARE");
      default:
        return std::string("Not implemented");
    }
  }
}