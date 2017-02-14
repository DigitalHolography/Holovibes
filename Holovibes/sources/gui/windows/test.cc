#include "main_window.hh"
#include <qstylefactory.h>

namespace gui
{
	void MainWindow::set_night()
	{
		theme_index_ = 1;
		qApp->setStyle(QStyleFactory::create("Fusion"));

		QPalette darkPalette;
		darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
		darkPalette.setColor(QPalette::WindowText, Qt::white);
		darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
		darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
		darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
		darkPalette.setColor(QPalette::ToolTipText, Qt::white);
		darkPalette.setColor(QPalette::Text, Qt::white);
		darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
		darkPalette.setColor(QPalette::ButtonText, Qt::white);
		darkPalette.setColor(QPalette::BrightText, Qt::red);
		darkPalette.setColor(QPalette::Disabled, QPalette::Text, Qt::darkGray);
		darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, Qt::darkGray);
		darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, Qt::darkGray);
		darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
		darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
		darkPalette.setColor(QPalette::HighlightedText, Qt::black);

		qApp->setPalette(darkPalette);

		qApp->setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }");
	}

	void MainWindow::set_classic()
	{
		theme_index_ = 0;
		qApp->setPalette(this->style()->standardPalette());
		qApp->setStyle(QStyleFactory::create("WindowsVista"));
		qApp->setStyleSheet("");
	}
}