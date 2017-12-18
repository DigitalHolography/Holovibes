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

/*! \file
*
* Wrapper around the vector of overlay. Permitting to manipulate all overlays of a window at once. */
#pragma once

#include "Overlay.hh"

namespace holovibes
{
	namespace gui
	{
		class OverlayManager
		{
		public:
			OverlayManager(BasicOpenGLWindow* parent);
			~OverlayManager();

			/*! \brief Create an overlay depending on the value passed to the template. */
			template <KindOfOverlay ko>
			void create_overlay();
			/*! \brief Create the default overlay in the view. Zoom for Direct/Holo, Cross for Slices. */
			void create_default();
			/*! \brief Create an overlay, and set its zone. */
			void set_zone(ushort frameside, units::RectFd zone, KindOfOverlay ko);

			/*! \brief Disable all the overlay of kind ko*/
			bool disable_all(KindOfOverlay ko);
			/*! \brief Disable all the overlays. If def is set, it will create a default overlay. */
			void reset(bool def = true);

			/*! \brief Call the press function of the current overlay. */
			void press(QMouseEvent *e);
			/*! \brief Call the keyPress function of the current overlay. */
			void keyPress(QKeyEvent *e);
			/*! \brief Call the move function of the current overlay. */
			void move(QMouseEvent *e);
			/*! \brief Call the release function of the current overlay. */
			void release(ushort frameSide);

			/*! \brief Draw every overlay that should be displayed. */
			void draw();
			/*! \brief Get the zone of the current overlay. */
			units::RectWindow getZone() const;
			/*! \brief Get the kind of the current overlay. */
			KindOfOverlay getKind() const;

			# ifdef _DEBUG
				/*! \brief Prints every overlay in the vector. Debug purpose. */
				void printVector();
			# endif

		private:
			//! Push in the vector the newly created overlay, set the current overlay, and call its init function.
			void create_overlay(std::shared_ptr<Overlay> new_overlay);

			/*! \brief Set the current overlay and notify observers to update gui. */
			void set_current(std::shared_ptr<Overlay> new_overlay);
			/*! \brief Try to set the current overlay to the first active overlay of a given type. */
			bool set_current(KindOfOverlay ko);

			/*! \brief Deletes from the vector every disabled overlay. */
			void clean();

			//! Containing every created overlay.
			std::vector<std::shared_ptr<Overlay>> overlays_;
			//! Current overlay used by the user.
			std::shared_ptr<Overlay> current_overlay_;

			/*! When we delete BasicOpenGlWindow which contains an instance of this,
			 * we cannot have a pointer to it otherwise it will never be destroyed.
			 * We could use weak_ptr instead of raw pointer. */
			BasicOpenGLWindow* parent_;
		};
	}
}
