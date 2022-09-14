#include <iomanip>
#include <sstream>
#include <Windows.h>
#include <filesystem>
#include <shlobj.h>

#include "tools.hh"
#include "logger.hh"
#include "tools_conversion.cuh"
#include "power_of_two.hh"

namespace holovibes
{
unsigned short upper_window_size(ushort width, ushort height) { return std::max(width, height); }

void get_good_size(ushort& width, ushort& height, ushort window_size)
{
    if (window_size == 0)
    {
        return;
    }

    if (width > height)
    {
        height = window_size * height / width;
        width = window_size;
    }
    else
    {
        width = window_size * width / height;
        height = window_size;
    }
}

std::string get_exe_dir()
{
#ifdef UNICODE
    wchar_t path[MAX_PATH];
#else
    char path[MAX_PATH];
#endif
    HMODULE hmodule = GetModuleHandle(NULL);
    if (hmodule != NULL)
    {
        GetModuleFileName(hmodule, path, (sizeof(path)));
        std::filesystem::path p(path);
        return p.parent_path().string();
    }

    Logger::main().error("Failed to find executable dir");
    throw std::runtime_error("Failed to find executable dir");
}

std::string get_record_filename(std::string filename)
{
    size_t dot_index = filename.find_last_of('.');
    if (dot_index == filename.npos)
        dot_index = filename.size();

    // Make sure 2 files don't have the same name by adding -1 / -2 / -3 ... in
    // the name
    unsigned i = 1;
    while (std::filesystem::exists(filename))
    {
        if (i == 1)
        {
            filename.insert(dot_index, "-1", 0, 2);
            ++i;
            continue;
        }
        unsigned digits_nb = std::log10(i - 1) + 1;
        filename.replace(dot_index, digits_nb + 1, "-" + std::to_string(i));
        ++i;
    }

    return filename;
}

QString create_absolute_qt_path(const std::string& relative_path)
{
    std::filesystem::path dir(get_exe_dir());
    dir = dir / relative_path;
    return QString(dir.string().c_str());
}

std::string create_absolute_path(const std::string& relative_path)
{
    std::filesystem::path dir(get_exe_dir());
    dir = dir / relative_path;
    return dir.string();
}

std::filesystem::path get_user_documents_path()
{
    wchar_t document_path[MAX_PATH];
    HRESULT sh_res = SHGetFolderPathW(0, CSIDL_MYDOCUMENTS, 0, 0, document_path);

    if (sh_res == S_OK)
    {
        char str[MAX_PATH];
        wcstombs(str, document_path, MAX_PATH - 1);
        return str;
    }

    return "";
}
} // namespace holovibes

std::string engineering_notation(double value, int nb_significant_figures)
{

    static std::string prefix[] = {"y", "z", "a", "f", "p", "n", "�", "m", "", "k", "M", "G", "T", "P", "E", "Z", "Y"};

    if (value == 0.)
    {
        return "0";
    }

    std::string res;

    if (value < 0)
    {
        res += '-';
        value = -value;
    }

    int expof10 = log10(value);
    if (expof10 > 0)
        expof10 = (expof10 / 3) * 3;
    else
        expof10 = (-expof10 + 3) / 3 * (-3);

    value *= pow(10, -expof10);

    std::string SI_prefix_symbol = prefix[expof10 / 3 + 8];

    int leading_figure = static_cast<int>(log10(value)) % 3 + 1;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(std::max(nb_significant_figures - leading_figure, 0)) << value << " "
       << SI_prefix_symbol;

    return ss.str();
}
