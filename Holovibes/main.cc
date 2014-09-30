#include "stdafx.h"
#include <iostream>
#include "option_parser.hh"
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>

using namespace holovibes;

int main(const int argc, const char *argv[])
{
  OptionParser *op = new OptionParser(argc, argv);
  op->init_parser();
  op->proceed();
  return 0;
}
