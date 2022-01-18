
#include "API.hh"
#include "MainWindow.hh"
#include "gui.hh"

namespace holovibes::gui
{
// FIXME API : filename is badly coded
void start_gui(int argc, char** argv, const std::string filename)
{
    holovibes::Holovibes::instance().is_cli = false;

    api::check_cuda_graphic_card();

    QLocale::setDefault(QLocale("en_US"));
    QApplication app(argc, argv);

    QSplashScreen splash(QPixmap(":/holovibes_logo.png"));
    splash.show();

    // Hide the possibility to close the console while using Holovibes
    HWND hwnd = GetConsoleWindow();
    HMENU hmenu = GetSystemMenu(hwnd, FALSE);
    EnableMenuItem(hmenu, SC_CLOSE, MF_GRAYED);

    // Create the window object that inherit from QMainWindow
    gui::MainWindow window;
    window.show();
    splash.finish(&window);

    // Set callbacks
    holovibes::GSH::instance().set_notify_callback([&]() { window.notify(); });
    holovibes::Holovibes::instance().set_error_callback([&](auto e) { window.notify_error(e); });

    if (!filename.empty())
        api::import_file(filename);

    // Resizing horizontally the window before starting
    window.layout_toggled();

    // Launch the Qt app
    if (app.exec())
        throw std::runtime_error("QT crash");
}
} // namespace holovibes::gui