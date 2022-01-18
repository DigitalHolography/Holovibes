#pragma once

class Dump
{
  public:
    template <typename T>
    void operator()(T& value)
    {
        LOG_DEBUG(main, "{} {} = {}", typeid(T).name(), value.get_key(), value);
    }
};