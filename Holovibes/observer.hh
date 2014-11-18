#ifndef OBSERVER_HH
# define OBSERVER_HH

namespace holovibes
{
  class Observer
  {
  public:
    virtual void notify() = 0;
  protected:
    Observer()
    {}

    virtual ~Observer()
    {}
  };
}

#endif /* !OBSERVER_HH */