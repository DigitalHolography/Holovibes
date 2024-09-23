#include <filesystem>
#include <fstream>
#include <iomanip>
#include <shlobj.h>
#include <sstream>
#include <Windows.h>

#include "chrono.hh"
#include "logger.hh"
#include "tools.hh"
#include "tools_conversion.cuh"

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

std::string get_record_filename(std::string filename)
{
    size_t dot_index = filename.find_last_of('.');
    if (dot_index == std::string::npos)
        dot_index = filename.size();

    // Make sure 2 files don't have the same name by adding _1 / _2 / _3 ... in
    // the name
    unsigned i = 1;

    auto name_index = filename.find_last_of('\\');
    if (name_index == std::string::npos)
    {
        // If no slash found, assume the last index before the dot is where to insert the date
        name_index = filename.find_last_of('.') - 1;
    }

    auto search = filename;
    search.insert(name_index + 1, Chrono::get_current_date() + "_");

    while (std::filesystem::exists(search))
    {
        if (i == 1)
        {
            search.insert(dot_index + 7, "_1", 0, 2); // + 7 because of the date
            ++i;
            continue;
        }
        unsigned digits_nb = static_cast<unsigned int>(std::to_string(i - 1).length());
        search.replace(dot_index + 7, digits_nb + 1, "_" + std::to_string(i));
        ++i;
    }
    i--;
    if (i == 0)
        return filename;
    return filename.insert(dot_index, "_" + std::to_string(i));
}

QString create_absolute_qt_path(const std::string& relative_path)
{
    std::filesystem::path dir(GET_EXE_DIR);
    dir = dir / relative_path;
    return QString(dir.string().c_str());
}

std::string create_absolute_path(const std::string& relative_path)
{
    std::filesystem::path dir(GET_EXE_DIR);
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
