cmake_minimum_required(VERSION 3.0)

# The headers that result in a moc_*.cpp are the only ones required in ths list
target_sources(Holovibes PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/gui/other/gui_curve_plot.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/gui/other/gui_drag_drop_lineedit.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/gui/other/gui_group_box.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/gui/windows/MainWindow.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/gui/windows/PlotWindow.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/recorder.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/thread/thread_csv_record.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/thread/thread_reader.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/thread/thread_recorder.hh"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/info_manager.hh"
)