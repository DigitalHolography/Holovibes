#pragma once

namespace holovibes::entities
{
struct BatchQuery
{
    unsigned int batch_size;
};

struct BatchCommand
{
    unsigned int batch_size;
};

struct TimeTranformationStrideQuery
{
    unsigned int time_transformation_stride;
};

struct TimeTranformationStrideCommand
{
    unsigned int time_transformation_stride;
};
} // namespace holovibes::entities
