; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

; CMake: build/Generator      VisualStudio: x64
#define BuildDir "build\Ninja\Release"

#define MyAppName "Holovibes"
#define MyAppVersion "8.4.1"
#define MyAppPublisher "Holovibes"
#define MyAppURL "http://www.holovibes.com/"
#define MyAppExeName "Holovibes.exe"

#define QtPath "C:\Qt\Qt5.9.0\5.9\msvc2017_64\bin"
#define QtPlatformPath "C:\Qt\Qt5.9.0\5.9\msvc2017_64\plugins\platforms"
#define CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
AppId={{905ACDE1-B1A5-43D3-A874-5C5FC50A47D8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={pf}\{#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=holovibes_setup_{#MyAppVersion}
Compression=lzma
SolidCompression=yes
DiskSpanning=no
UninstallDisplayName=Holovibes
UninstallDisplayIcon={app}\{#MyAppVersion}\Holovibes.exe
UninstallFilesDir={app}\{#MyAppVersion}
SetupIconFile="{#BuildDir}\Holovibes.ico"

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 0,6.1

[Types]
Name: "Full"; Description: "Full installation"
Name: "Compact"; Description: "Compact installation"
Name: "Custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "program"; Description: "Holovibes"; Types: full compact custom; Flags: fixed
Name: "visual"; Description: "Run-time components for C++"; Types: full

[Files]
Source: "{#BuildDir}\*"; DestDir: "{app}\{#MyAppVersion}"; Components: program; Flags: ignoreversion recursesubdirs

Source: "{#QtPath}\Qt5Core.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Gui.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5OpenGL.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5PrintSupport.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Widgets.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Svg.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Charts.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPlatformPath}\*"; DestDir: "{app}\{#MyAppVersion}\platforms";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\cufft64_10.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\cublas64_11.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\cublasLt64_11.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\cusolver64_11.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\nppc64_11.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\nppial64_11.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\nppist64_11.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion

Source: "resources\setup_creator_files\vcredist_2019_x64.exe"; DestDir: "{tmp}"; Components: visual; Flags: nocompression ignoreversion;

[UninstallDelete]
Type: filesandordirs; Name: "{app}\{#MyAppVersion}"

[Run]
Filename: "{tmp}\vcredist_2019_x64.exe"; Parameters: "/install /passive /norestart"; Components: visual; Flags: waituntilterminated

[Icons]
Name: "{commonprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppVersion}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppVersion}\{#MyAppExeName}"; Tasks: desktopicon