/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "cli.hh"

#include <chrono>

#include "tools.hh"
#include "icompute.hh"
#include "ini_config.hh"
#include "input_frame_file_factory.hh"

namespace cli
{
static void progress_bar(int current, int total, int length)
{
    std::string text;
    text.reserve(length + 2);

    float ratio = (current * 1.0f) / total;
    int n = length * ratio;

    text += '[';
    if (n == length)
    {
        text.append(n, '=');
    }
    else if (n > 0)
    {
        text.append(n - 1, '=');
        text.append(1, '>');
    }
    text.append(length - n, ' ');
    text += ']';

    std::cout << '\r' << text;
    std::cout.flush();
}

static void print_verbose(const holovibes::OptionsDescriptor& opts)
{
    std::cout << "Config:\n";
    auto ini_data =
        read_file<std::string>(opts.ini_path.value_or(GLOBAL_INI_PATH));
    std::cout << ini_data << "\n\n";

    std::cout << "Input file: " << opts.input_path.value() << "\n";
    std::cout << "Output file: " << opts.output_path.value() << "\n";
    std::cout << "FPS: " << opts.fps.value_or(60) << "\n";
    std::cout << "Number of frames to record: ";
    if (opts.n_rec)
        std::cout << opts.n_rec.value() << "\n";
    else
        std::cout << "full file\n";
    std::cout << "Raw recording: " << std::boolalpha << opts.record_raw
              << std::dec << "\n";

    std::cout << std::endl;
}

int start_cli(holovibes::Holovibes& holovibes,
              const holovibes::OptionsDescriptor& opts)
{
    if (opts.verbose)
    {
        print_verbose(opts);
    }

    std::string ini_path = opts.ini_path.value_or(GLOBAL_INI_PATH);
    holovibes::ini::load_ini(holovibes.get_cd(), ini_path);
    holovibes.start_information_display(true);

    std::string input_path = opts.input_path.value();

    holovibes::io_files::InputFrameFile* input_frame_file =
        holovibes::io_files::InputFrameFileFactory::open(input_path);

    const camera::FrameDescriptor& fd =
        input_frame_file->get_frame_descriptor();
    size_t input_nb_frames = input_frame_file->get_total_nb_frames();

    const unsigned int fps = opts.fps.value_or(60);
    holovibes.init_input_queue(fd);
    holovibes.start_file_frame_read(input_path,
                                    true,
                                    fps,
                                    0,
                                    input_nb_frames,
                                    false);

    input_frame_file->import_compute_settings(holovibes.get_cd());

    holovibes.get_cd().compute_mode = holovibes::Computation::Hologram;
    holovibes.get_cd().frame_record_enabled = true;

    holovibes.start_compute();

    holovibes.get_compute_pipe()->request_refresh();

    holovibes.start_frame_record(opts.output_path.value(),
                                 opts.n_rec.value_or(input_nb_frames),
                                 opts.record_raw,
                                 false);

    const auto& info = holovibes.get_info_container();
    auto progress_opt = info.get_progress_index(
        holovibes::InformationContainer::ProgressType::FRAME_RECORD);

    auto begin = std::chrono::steady_clock::now();

    while (holovibes.get_cd().frame_record_enabled)
    {
        if (!progress_opt)
            progress_opt = info.get_progress_index(
                holovibes::InformationContainer::ProgressType::FRAME_RECORD);
        else
        {
            const auto& progress = progress_opt.value();
            progress_bar(progress.first->load(), progress.second->load(), 40);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    progress_bar(1, 1, 40); // show 100% completion to avoid rounding errors

    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count();

    printf(" Time: %.3fs\n", duration / 1000.0f);

    holovibes.stop_all_worker_controller();

    delete input_frame_file;

    return 0;
}
} // namespace cli
