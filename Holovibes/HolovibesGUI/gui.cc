
#include "API.hh"
#include "gui.hh"
#include "MainWindow.hh"

#include "gui_front_end.hh"

namespace holovibes::gui
{
void start_gui(int argc, char** argv, const std::string filename)
{
    ComputeCacheFrontEndMethods::link_front_end<GuiFrontEndForComputeCacheOnPipeRequest>();
    ImportCacheFrontEndMethods::link_front_end<GuiFrontEndForImportCacheOnPipeRequest>();
    ExportCacheFrontEndMethods::link_front_end<GuiFrontEndForExportCacheOnPipeRequest>();
    ViewCacheFrontEndMethods::link_front_end<GuiFrontEndForViewCacheOnPipeRequest>();
    AdvancedCacheFrontEndMethods::link_front_end<GuiFrontEndForAdvancedCacheOnPipeRequest>();
    api::detail::set_value<FrontEnd>("HolovibesGUI");

    api::detail::set_value<ExportRecordDontLoseFrame>(false);

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

    if (filename.empty() == false)
        api::detail::set_value<ImportFilePath>(filename);

    // Resizing horizontally the window before starting
    window.layout_toggled();

    // Launch the Qt app
    if (app.exec())
        throw std::runtime_error("QT crash");
}

} // namespace holovibes::gui
