#include "real_position.hh"
#include "fd_pixel.hh"

namespace holovibes::units {
	RealPosition::RealPosition(ConversionData data, Axis axis, double val)
		: Unit(data, axis, val)
	{}
}
