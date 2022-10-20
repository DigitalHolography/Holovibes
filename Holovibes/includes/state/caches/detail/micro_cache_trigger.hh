#pragma once

#include <functional>
#include <memory>
#include <iostream>

namespace holovibes
{

template <typename T>
class TriggerChangeValue
{
  public:
    T* value_;
    std::function<void(void)>& callback_;

  public:
    TriggerChangeValue(std::function<void(void)>& callback, T* value)
        : value_(value)
        , callback_(callback)
    {
        std::cout << "CREATE TRIGGER" << std::endl;
    }

    ~TriggerChangeValue()
    {
        std::cout << "DESTROY TRIGGER" << std::endl;
        callback_();
    }

  public:
    T* operator->() { return value_; }
};

} // namespace holovibes
