#include <windows.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Quote an argument for the PowerShell invocation.
std::wstring QuoteArg(const std::wstring& arg) {
    if (arg.find_first_of(L" \t\"") == std::wstring::npos) {
        return arg;
    }

    std::wstring quoted;
    quoted.reserve(arg.size() + 2);
    quoted.push_back(L'"');
    for (wchar_t ch : arg) {
        if (ch == L'"') {
            quoted.push_back(L'\\');
        }
        quoted.push_back(ch);
    }
    quoted.push_back(L'"');
    return quoted;
}

}  // namespace

int wmain(int argc, wchar_t* argv[]) {
    const std::filesystem::path exePath = std::filesystem::path(argv[0]).parent_path();
    const std::filesystem::path scriptPath = exePath / L"ProcessHoloFiles.ps1";

    if (!std::filesystem::exists(scriptPath)) {
        std::wcerr << L"[ProcessHoloFiles] Unable to locate \"" << scriptPath.wstring() << L"\"." << std::endl;
        return 1;
    }

    std::filesystem::path powerShellPath;
    wchar_t systemRoot[MAX_PATH];
    const DWORD rootLen = GetEnvironmentVariableW(L"SystemRoot", systemRoot, MAX_PATH);
    if (rootLen > 0 && rootLen < MAX_PATH) {
        powerShellPath = std::filesystem::path(systemRoot) / L"System32/WindowsPowerShell/v1.0/powershell.exe";
    }
    const std::wstring powerShellExe = std::filesystem::exists(powerShellPath) ? powerShellPath.wstring() : L"powershell.exe";

    std::wstring commandLine = QuoteArg(powerShellExe) + L" -NoProfile -ExecutionPolicy Bypass -File " + QuoteArg(scriptPath.wstring());
    for (int i = 1; i < argc; ++i) {
        commandLine.push_back(L' ');
        commandLine += QuoteArg(argv[i]);
    }

    std::vector<wchar_t> mutableCommand(commandLine.begin(), commandLine.end());
    mutableCommand.push_back(L'\0');

    STARTUPINFOW startupInfo{};
    startupInfo.cb = sizeof(startupInfo);
    PROCESS_INFORMATION processInfo{};

    const BOOL started = CreateProcessW(
        nullptr,
        mutableCommand.data(),
        nullptr,
        nullptr,
        FALSE,
        0,
        nullptr,
        exePath.c_str(),
        &startupInfo,
        &processInfo);

    if (!started) {
        std::wcerr << L"[ProcessHoloFiles] Failed to start PowerShell ("
                   << powerShellExe << L"). Error: " << GetLastError() << std::endl;
        return 1;
    }

    WaitForSingleObject(processInfo.hProcess, INFINITE);
    DWORD exitCode = 0;
    GetExitCodeProcess(processInfo.hProcess, &exitCode);

    CloseHandle(processInfo.hThread);
    CloseHandle(processInfo.hProcess);
    return static_cast<int>(exitCode);
}
