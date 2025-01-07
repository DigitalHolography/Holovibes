#include <filesystem>
#include <fstream>
#include <iomanip>
#include <shlobj.h>
#include <sstream>
#include <Windows.h>

#include "API.hh"
#include "chrono.hh"
#include "logger.hh"
#include "tools.hh"
#include "tools_conversion.cuh"

namespace holovibes
{
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

std::string get_record_filename(std::string file_path, std::string append, std::string prepend)
{
    std::filesystem::path filePath(file_path);
    std::string filename = filePath.filename().stem().string();
    std::string extension = filePath.extension().string();
    std::string path = filePath.parent_path().string();
    std::filesystem::path oldFilePath = path + "/" + prepend + "_" + filename + append;
    std::filesystem::path newFilePath = oldFilePath.string() + extension;

    for (int i = 1; std::filesystem::exists(newFilePath); ++i)
    {
        newFilePath = oldFilePath.string() + "_" + std::to_string(i) + extension;
    }

    return newFilePath.string();
}

QString create_absolute_qt_path(const std::string& relative_path)
{
    std::filesystem::path dir(GET_EXE_DIR);
    dir = dir / relative_path;
    return QString(dir.string().c_str());
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