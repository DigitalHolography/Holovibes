#pragma once

#include <exception>
#include <string>

namespace gpib
{
class GpibBadAlloc : public std::exception
{
  public:
    virtual ~GpibBadAlloc() {}

    virtual const char* what() const override { return "Failed to allocate memory for VISA interface"; }
};

class GpibInvalidPath : public std::exception
{
  public:
    GpibInvalidPath(const std::string& path)
        : msg_("Could not open file : " + path)
    {
    }

    virtual ~GpibInvalidPath() {}

    virtual const char* what() const override { return msg_.c_str(); }

  private:
    const std::string msg_;
};

class GpibNoFilepath : public std::exception
{
  public:
    virtual ~GpibNoFilepath() {}

    virtual const char* what() const override { return "No filepath provided"; }
};

class GpibParseError : public std::exception
{
  public:
    enum ErrorType
    {
        NoBlock,
        NoAddress,
        NoWait
    };

    GpibParseError(size_t line, const ErrorType type)
    {
        msg_ = "Bad format at line ";
        msg_.append(std::to_string(line));

        if (type == NoBlock)
            msg_.append(" : no #Block");
        if (type == NoAddress)
            msg_.append(" : no valid instrument address");
        if (type == NoWait)
            msg_.append(" : no valid wait status");
    }

    virtual ~GpibParseError() {}

    virtual const char* what() const override { return msg_.c_str(); }

  private:
    std::string msg_;
};

class GpibSetupError : public std::exception
{
  public:
    virtual ~GpibSetupError() {}

    virtual const char* what() const override { return "Could not setup VISA driver"; }
};

class GpibInstrError : public std::exception
{
  public:
    GpibInstrError(const std::string& address)
        : msg_("Could not setup connexion at address " + address)
    {
    }

    virtual ~GpibInstrError() {}

    virtual const char* what() const override { return msg_.c_str(); }

  private:
    const std::string msg_;
};

class GpibBlankFileError : public std::exception
{
  public:
    virtual ~GpibBlankFileError() {}

    virtual const char* what() const override { return "The batch file is empty!"; }
};
} // namespace gpib
