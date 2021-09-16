#include "checker.hh"

Checker* Checker::instance()
{
    static Checker checker;
    return &checker;
}

CheckLine Checker::check__(const bool cond_res, const std::string& cond_str)
{
    return CheckLine(cond_res, cond_str);
}