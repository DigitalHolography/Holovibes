#pragma once

#include <string>
#define LOGURU_WITH_STREAMS 1

#include <exception>
#include <fstream>
#include <sstream>
#include <functional>
#include <iostream>

#include "loguru.hpp"

#define CHECK(cond) Checker::instance()->check__(cond, #cond)

class CheckLine
{
  public:
    CheckLine(const bool cond_res, const std::string& cond_str)
        : cond_res_(cond_res)
        , cond_str_(cond_str)
    {
    }

    template <typename T>
    CheckLine& operator<<(const T& data)
    {
        buffer << data;
        return *this;
    }

    template <typename CharT, typename Traits>
    CheckLine& operator<<(const std::basic_ostream<CharT, Traits>& (*endl)(
        std::basic_ostream<CharT, Traits>&))
    {
        buffer << endl;
        return *this;
    }

    ~CheckLine()
    {
        if (!cond_res_)
        {
            if (!buffer.str().empty())

                DCHECK_F(false,
                         "[%s] : %s",
                         cond_str_.c_str(),
                         buffer.str().c_str());
            else
                DCHECK_F(false, "[%s]", cond_str_.c_str());
        }
    }

  private:
    const bool cond_res_;
    const std::string cond_str_;
    std::stringstream buffer;
};

class Checker
{
  public:
    static Checker* instance();

    CheckLine check__(const bool cond_res, const std::string& cond);

  private:
    Checker() = default;
};