/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

namespace holovibes
{
void device_print(uchar* d_data, size_t offset, size_t nb_elts);

void device_print(ushort* d_data, size_t offset, size_t nb_elts);

void device_print(float* d_data, size_t offset, size_t nb_elts);

void device_print(cuComplex* d_data, size_t offset, size_t nb_elts);
} // namespace holovibes
