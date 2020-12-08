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

#include "stdafx.hh"
#include "gui_drag_drop_lineedit.hh"

namespace holovibes::gui {

	Drag_drop_lineedit::Drag_drop_lineedit(QWidget * parent)
		: QLineEdit(parent)
	{
		setPlaceholderText("Drop file here");
	}

	void Drag_drop_lineedit::dropEvent(QDropEvent * event)
	{
		auto url = event->mimeData()->urls()[0];
		auto path = url.path();
		if (path.at(0) == '/')
			path.remove(0, 1);
		setText(path);
	}
}
