#pragma once

#ifdef GPIB_EXPORTS
# define GPIB_API __declspec(dllexport)
#else
# define GPIB_API __declspec(dllimport)
#endif

#include <string>

namespace gpib
{
  /*! \brief Interface in order to load the dll runtime
   *
   * We want holovibes to run even if the computer doesn't have visa
   * installed on it (it is very heavy and long to install). So what we
   * want is to load visa's dll only when needed (batch input).
   *
   * This interface only has got these methods because they are the one
   * used in holovibes. The rest of the methods are only used and GPIB
   * and don't need to be in there.
   *
   * Why do we use this interface ?
   *
   * Because otherwhise, holovibes would be faced to some unrecognized symbols
   */
  class IVisaInterface
  {
  public:
    /*!
      I tried to mimic icamera.hh so this will need some improvement
      in the futur
    */

    IVisaInterface()
    {}

    IVisaInterface(const std::string path);

    virtual ~IVisaInterface()
    {}

    virtual bool execute_next_block() = 0;

  };

  /* \brief See icamera.hh to have more information about this */
  extern "C"
  {
    GPIB_API IVisaInterface* new_gpib_controller(const std::string path);
  }
}