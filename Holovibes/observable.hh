#ifndef OBSERVABLE_HH
# define OBSERVABLE_HH

# include <vector>

# include "observer.hh"

namespace holovibes
{
  class Observable
  {
  public:
    void register_observer(Observer& o)
    {
      observers_.push_back(&o);
    }

    void notify_observers()
    {
      for (std::vector<Observer*>::iterator it = observers_.begin();
        it != observers_.end();
        ++it)
        (*it)->notify();
    }

  protected:
    Observable()
    {}

    virtual ~Observable()
    {}

  private:
    std::vector<Observer*> observers_;
  };
}

#endif /* !OBSERVABLE_HH */