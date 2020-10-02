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

/*! \file
 *
 * Interface for a thread encapsulation class that
 * grabs images from a source. */
#pragma once

#include <string>

/*! Forward declaration*/
namespace camera
{
  struct FrameDescriptor;
}

namespace holovibes
{
    //If you can find better names, please replace them with yours
  enum class SquareInputMode
  {
    NO_MODIFICATION,
    ZERO_PADDED_SQUARE,
    CROPPED_SQUARE
  };

  SquareInputMode get_square_input_mode_from_string(const std::string &name);
  std::string get_string(const SquareInputMode mode);


  /*! \brief Interface for a thread encapsulation class that
   * grabs images from a source.
   */
  class IThreadInput
  {
  public:
    virtual ~IThreadInput();

    virtual const camera::FrameDescriptor& get_input_fd() const = 0;

    //If the input is not square, it might need to be croped or embedded into one
    //This would be the effective frame descriptor for the rest of the program
    virtual const camera::FrameDescriptor& get_queue_fd() const = 0;
  public:
    /*! \brief Stop thread and join it */
    bool stop_requested_;

  protected:
    IThreadInput();
  };
}