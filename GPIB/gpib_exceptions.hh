/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

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