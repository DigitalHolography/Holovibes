#include "stdafx.hh"
#include "gui_drag_drop_lineedit.hh"

namespace holovibes::gui {

	Drag_drop_lineedit::Drag_drop_lineedit(QWidget * parent)
		: QLineEdit(parent)
	{
		setAcceptDrops(true);
		setDragEnabled(false);
	}

	void Drag_drop_lineedit::dropEvent(QDropEvent * event)
	{
		//const QMimeData* mimeData = event->mimeData();
		auto url = event->mimeData()->urls()[0];
		auto path = url.path();
		if (path.at(0) == '/')
			path.remove(0, 1);
		setText(path);
	}

	/*void Drag_drop_lineedit::dragMoveEvent(QDragMoveEvent event)
	{
		const QMimeData* mimeData = event->mimeData();
	}*/
}

//#include "moc_gui_drag_drop_lineedit.cc"