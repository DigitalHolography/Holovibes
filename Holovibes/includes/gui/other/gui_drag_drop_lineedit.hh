#pragma once

namespace holovibes::gui {
	class Drag_drop_lineedit : public QLineEdit
	{
		Q_OBJECT
	public:
		Drag_drop_lineedit(QWidget* parent = nullptr);
		
		public slots:
		void dropEvent(QDropEvent* event) override;
		void dragEnterEvent(QDragEnterEvent* e) override {
			e->acceptProposedAction();
		}
	};
}
