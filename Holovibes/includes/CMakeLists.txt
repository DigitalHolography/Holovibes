# The headers that result in a moc_*.cpp are the only ones required in ths list
target_sources("${PROJECT_NAME}_lib" PRIVATE
    gui/other/gui_curve_plot.hh
    gui/other/gui_drag_drop_lineedit.hh
    gui/other/gui_group_box.hh
    gui/other/gui_frame.hh
    gui/windows/MainWindow.hh
    gui/windows/PlotWindow.hh
    gui/windows/mainWindow/panel.hh
    gui/windows/mainWindow/import_panel.hh
    gui/windows/mainWindow/export_panel.hh
)
