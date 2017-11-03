/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#pragma once

#include "frame_desc.hh"
#include "units/conversion_data.hh"

namespace holovibes
{
	namespace units
	{
		/*! \brief A generic distance unit type
		 *
		 * Used to define implicit conversions between the different units
		 * T will be either float or int, defined in the child classes
		 */
		template<typename T>
		class Unit
		{
		public:

			using primary_type = T;

			Unit(ConversionData data, T val)
				: conversion_data_(data)
				, val_(val)
			{}

			/*! \brief Implicit cast toward the primary type
			 */
			operator T() const
			{
				return val_;
			}

			/*! \brief Implicit cast toward the primary type
			 */
			operator T&()
			{
				return val_;
			}

		protected:
			/*! \brief Encapsulates the metadata needed for the conversions
			 */
			ConversionData	conversion_data_;

			/*! \brief The value itself
			 */
			T				val_;
		};
	}
}