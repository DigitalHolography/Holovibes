/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Various fast functions to find powers of two. */
#pragma once

/* \brief check if x is power of two
 * http://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c
 * http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
 */
bool isPowerOfTwo(unsigned int x);

/*! \brief Return the next power of two */
unsigned int nextPowerOf2(unsigned int x);

/*! \brief Return the previous power of two */
unsigned int prevPowerOf2(unsigned int x);