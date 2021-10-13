#pragma once

#define NAMEOF(name) #name

template <typename T>
class MicroCache
{
    T data;
    std::shared_ptr<std::unordered_map<std::string, bool>> updated_data;

    void synchronize();
};

template <typename T>
class Monitored
{
    T value;
    std::shared_ptr<std::unordered_map<std::string, bool>> updated_data;

    Monitored& operator=(T other)
    {
        value = other;
        updated_data[NAMEOF(this)] = true;
        return *this;
    }
};
