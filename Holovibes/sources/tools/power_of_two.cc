#include "power_of_two.hh"

bool isPowerOfTwo(unsigned int x)
{
  return ((x != 0) && ((x & (~x + 1)) == x));
}

unsigned int nextPowerOf2(unsigned int x)
{
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  ++x;
  return (x);
}

unsigned int prevPowerOf2(unsigned int x)
{
  return nextPowerOf2(x - 1) >> 1;
}