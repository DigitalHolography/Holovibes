/*! \file
 *
 * \brief Implementation of the VectorJob class
 */
#pragma once

#include "jobs/job.hh"

namespace holovibes
{

/*!
 * \brief Parent class for all output job, be it in a queue, or just a buffer
 *
 */
class OutputJob : public Job
{
    operator std::string() const override { return "OutputJob{}"; }
};
} // namespace holovibes