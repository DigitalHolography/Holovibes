/*! \file
 *
 * \brief This files include the `device_print` function that prints the content of an array stored on the GPU.
 */
#pragma once

namespace holovibes
{

/*!
 * \brief      Print on stdout an array stored in the GPU memory
 *
 * The array is printed on a single line and values are separated by spaces.
 *
 * \param      d_data   The array to print
 * \param[in]  offset   The offset
 * \param[in]  nb_elts  The number of elements to print
 *
 * \tparam     T        The type of the array
 */
template <typename T>
void device_print(T* d_data, size_t offset, size_t nb_elts);

/*!
 * \brief      Print on stdout an array of complex stored in the GPU memory
 *
 * The array is printed on a single line and values are separated by pipe (|).
 * Each complex are written with the real part followed by a space and the imaginiray part.
 *
 * \param      d_data   The complex array to print
 * \param[in]  offset   The offset
 * \param[in]  nb_elts  The number of elements to print
 */
void device_print(cuComplex* d_data, size_t offset, size_t nb_elts);

} // namespace holovibes
