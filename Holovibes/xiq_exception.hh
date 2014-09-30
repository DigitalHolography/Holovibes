#ifndef XIQ_EXCEPTION_HH
# define XIQ_EXCEPTION_HH

# include <string>
# include <xiApi.h>

# include "camera_exception.hh"

namespace camera
{
  /*! This class is a wrapper of Xiq API error codes. */
  class XiqException : public CameraException
  {
  public:
    XiqException(XI_RETURN code)
      : CameraException("XiQ")
      , code_(code)
    {}

    virtual const char* what() const override;
  private:
    /*! Match Xiq API error code to return an error message. */
    std::string match_error(XI_RETURN code) const;
  private:
    XI_RETURN code_;
  };
}

#endif /* !XIQ_EXCEPTION_HH */