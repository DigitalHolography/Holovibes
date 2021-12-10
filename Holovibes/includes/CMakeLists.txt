# The headers that result in a moc_*.cpp are the only ones required in this list
target_sources("${PROJECT_NAME}_lib" PRIVATE
    gui/other/gui_curve_plot.hh
    gui/other/gui_drag_drop_lineedit.hh
    gui/other/gui_group_box.hh
    gui/other/gui_frame.hh
    gui/windows/MainWindow.hh
    gui/windows/AdvancedSettingsWindow.hh
    gui/windows/PlotWindow.hh
    gui/windows/panels/panel.hh
    gui/windows/panels/import_panel.hh
    gui/windows/panels/export_panel.hh
    gui/windows/panels/info_panel.hh
    gui/windows/panels/view_panel.hh
    gui/windows/panels/composite_panel.hh
    gui/windows/panels/image_rendering_panel.hh

    gui/windows/AdvancedSettingsWindowTools/q_double_spin_box_layout.hh
    gui/windows/AdvancedSettingsWindowTools/q_int_spin_box_layout.hh

    gui/windows/panels/asw_mainwindow_panel.hh
)
