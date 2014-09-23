#ifndef ERRORS_ENUM_HH
# define ERRORS_ENUM_HH

namespace error
{
  typedef enum errors
  {
#define ERR_MSG(Code, Msg) Code,
    // Here all .def includes
#include "err\wnd_gl.def"
#undef ERR_MSG
  } e_errors;
}

#endif /* !ERRORS_ENUM_HH */