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
      return "failed to allocate memory for VISA interface";
    }
  };

  class GpibInvalidPath : public std::exception
  {
  public:
    GpibInvalidPath(const std::string& path)
      : path_ { path }
    {
    }

    virtual ~GpibInvalidPath()
    {
    }

    virtual const char* what() const override
    {
      std::string msg("could not open file : ");
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
      return "no filepath provided";
    }
  };

  class GpibParseError : public std::exception
  {
  public:
    GpibParseError(const std::string& path)
      : path_ { path }
    {
    }

    virtual ~GpibParseError()
    {
    }

    virtual const char* what() const override
    {
      std::string msg("bad command format");
      msg.append(path_);
      return msg.c_str();
    }

  private:
    const std::string path_;
  };

  class GpibSetupError : public std::exception
  {
  public:
    virtual ~GpibSetupError()
    {
    }

    virtual const char* what() const override
    {
      return "could not setup VISA driver";
    }
  };

  class GpibInstrError : public std::exception
  {
  public:
    GpibInstrError(const std::string& address)
      : address_ { address }
    {
    }

    virtual ~GpibInstrError()
    {
    }

    virtual const char* what() const override
    {
      std::string msg("could not setup connexion at address ");
      msg.append(address_);
      return msg.c_str();
    }

  private:
    const std::string address_;
  };
}