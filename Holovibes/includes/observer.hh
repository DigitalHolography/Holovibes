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

# include <exception>
/*! \file
 *
 * Implementation of the Observer design pattern. */
#pragma once

namespace holovibes
{
  /*! \brief Implementation of the Observer design pattern.
   *
   * Set a child class to be observer of an observable object. */
  class Observer
  {
  public:
    /*! \brief Notify method called when Observable class change.
     *
     * Mandatory method that is called when an Observable object has changed of
     * state.
     */
    virtual void notify() = 0;

	virtual void notify_error(std::exception& e, const char* msg) = 0;

  protected:
    Observer()
    {
    }

    virtual ~Observer()
    {
    }
  };
}