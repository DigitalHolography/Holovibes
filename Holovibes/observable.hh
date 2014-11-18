#ifndef OBSERVABLE_HH
# define OBSERVABLE_HH

# include <set>
# include <algorithm>

# include "observer.hh"

namespace holovibes
{
  class Observable
  {
  public:
    void register_observer(Observer& o)
    {
      observers_.insert(o);
    }

    void unregister_observer(Observer& o)
    {
      observers_.erase(o);
    }

    void notify_observers()
    {
      for (auto it = observers_.begin();
        it != observers_.end();
        ++it)
        (*it).notify();
    }

  protected:
    Observable()
    {}

    virtual ~Observable()
    {}

  private:
    std::set<Observer&> observers_;
  };
}

#endif /* !OBSERVABLE_HH */