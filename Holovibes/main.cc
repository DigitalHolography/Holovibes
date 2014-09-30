#include "stdafx.h"
#include <iostream>
#include "option_parser.hh"
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>


namespace po = boost::program_options;


int main(int argc, char *argv[])
{
    Option_parser *op = new Option_parser(argc, argv);
    op->init_parser();
    op->proceed();
  return 0;
}