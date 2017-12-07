#pragma once

namespace holovibes::gui {
	class Drag_drop_lineedit : public QLineEdit
	{
		Q_OBJECT
	public:
		Drag_drop_lineedit(QWidget* parent = nullptr);
			
		//Q_SIGNALS:
		public slots:
		void dropEvent(QDropEvent* event) override;
		void dragEnterEvent(QDragEnterEvent* e) override {
			e->acceptProposedAction();
			/*if (e->mimeData()->hasFormat("application/x-qabstractitemmodeldatalist")) {
				e->acceptProposedAction();
			}*/
		}
		//void dragMoveEvent(QDragMoveEvent event) override;
	};
}
