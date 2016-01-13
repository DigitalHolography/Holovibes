#pragma once

# include <exception>
# include <string>

namespace gpib
{
  class GpibBadAlloc : public std::exception
  {
  public:
    virtual ~GpibBadAlloc()
    {
    }

    virtual const char* what() const override
    {
      return "Failed to allocate memory for VISA interface";
    }
  };

  class GpibInvalidPath : public std::exception
  {
  public:
    GpibInvalidPath(const std::string& path)
      : path_(path)
    {
    }

    virtual ~GpibInvalidPath()
    {
    }

    virtual const char* what() const override
    {
      std::string msg("Could not open file : ");
      msg.append(path_);
      return msg.c_str();
    }

  private:
    const std::string path_;
  };

  class GpibNoFilepath : public std::exception
  {
  public:
    virtual ~GpibNoFilepath()
    {
    }

    virtual const char* what() const override
    {
      return "No filepath provided";
    }
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

    GpibParseError(const std::string& line, const ErrorType type)
      : line_(line)
      , type_ { type }
    {
    }

    virtual ~GpibParseError()
    {
    }

    virtual const char* what() const override
    {
      std::string msg("Bad format at line ");
      msg.append(line_);

      if (type_ == NoBlock)
        msg.append(" : no #Block");
      if (type_ == NoAddress)
        msg.append(" : no valid instrument address");
      if (type_ == NoWait)
        msg.append(" : no valid wait status");

      return msg.c_str();
    }

  private:
    const std::string line_;
    const ErrorType type_;
  };

  class GpibSetupError : public std::exception
  {
  public:
    virtual ~GpibSetupError()
    {
    }

    virtual const char* what() const override
    {
      return "Could not setup VISA driver";
    }
  };

  class GpibInstrError : public std::exception
  {
  public:
    GpibInstrError(const std::string& address)
      : address_(address)
    {
    }

    virtual ~GpibInstrError()
    {
    }

    virtual const char* what() const override
    {
      std::string msg("Could not setup connexion at address ");
      msg.append(address_);
      return msg.c_str();
    }

  private:
    const std::string address_;
  };

  class GpibBlankFileError : public std::exception
  {
  public:
    virtual ~GpibBlankFileError()
    {
    }

    virtual const char* what() const override
    {
      return "The batch file is empty!";
    }
  };
}