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
 * Implementation of the Observable design pattern. */
#pragma once

# include <algorithm>
# include <vector>
# include "observer.hh"

namespace holovibes
{
  /*! \brief Implementation of the Observable design pattern.
   *
   * Set a child class to be Observable and contains a list of Observer that
   * will be notified when the Observable class is modified. */
  class Observable
  {
  public:
    /*! \brief add Observer in list*/
    void register_observer(Observer& o)
    {
      observers_.push_back(&o);
    }

    /*! \brief notify all Observer in list */
    void notify_observers()
    {
		std::for_each(observers_.begin(),
        observers_.end(),
        [](Observer* observer) { observer->notify(); });
    }

	/*! \brief notify all Observer in list that an error occured */
	void notify_error_observers(std::exception& e)
	{
		std::for_each(observers_.begin(),
			observers_.end(),
			[&](Observer* observer) { observer->notify_error(e); });
	}

  protected:
    Observable()
    {
    }

    virtual ~Observable()
    {
    }

  private:
    std::vector<Observer*> observers_;
  };
}