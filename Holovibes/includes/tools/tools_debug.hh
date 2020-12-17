/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

namespace holovibes
{
void device_print(uchar* d_data, size_t offset, size_t nb_elts);

void device_print(ushort* d_data, size_t offset, size_t nb_elts);

void device_print(float* d_data, size_t offset, size_t nb_elts);

void device_print(cuComplex* d_data, size_t offset, size_t nb_elts);
} // namespace holovibes