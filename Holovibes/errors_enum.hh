#ifndef ERRORS_ENUM_HH
# define ERRORS_ENUM_HH

namespace error
{
  typedef enum errors
  {
#define ERR_MSG(code, msg) code,
// Here all .def includes
#include "errors_pike_camera.def"
#undef ERR_MSG
  } e_errors;
}

#endif /* !ERRORS_ENUM_HH */