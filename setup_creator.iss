; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "Holovibes"
#define MyAppVersion "4.0.0"
#define MyAppPublisher "CNRS"
#define MyAppURL "http://www.holovibes.com/"
#define MyAppExeName "Holovibes.exe"
;define for Qt and Qwt path
#define QtPath "C:\Qt\Qt5.5.0\5.5\msvc2013_64\bin"
#define QtPlatformPath "C:\Qt\Qt5.5.0\5.5\msvc2013_64\plugins\platforms"
#define QwtPath "C:\Qt\qwt-6.1.2\lib"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{905ACDE1-B1A5-43D3-A874-5C5FC50A47D8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={pf}\{#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=setup
Compression=lzma
SolidCompression=yes
DiskSpanning=yes
UninstallDisplayName=Holovibes
UninstallDisplayIcon={app}\Holovibes.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 0,6.1

[Files]
Source: "x64\Release\Holovibes.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\atmcd64d.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\CameraAdimec.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\CameraIds.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\CameraIxon.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\CameraPCOEdge.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\CameraPCOPixelfly.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\CameraPike.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\CameraUtils.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\CameraXiq.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\FGCamera.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\GPIB.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\m3apiX64.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\SC2_Cam.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\uEye_api_64.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\edge.ini"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\ids.ini"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\ixon.ini"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\pike.ini"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\pixelfly.ini"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\xiq.ini"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\icon1.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "x64\Release\holovibes_logo.png"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#QwtPath}\qwt.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#QwtPath}\qwtd.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#QtPath}\Qt5Core.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#QtPath}\Qt5Gui.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#QtPath}\Qt5OpenGL.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#QtPath}\Qt5PrintSupport.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#QtPath}\Qt5Widgets.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#QtPath}\Qt5Svg.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#QtPlatformPath}\*"; DestDir: "{app}\platforms"; Flags: ignoreversion
Source: "setup_creator_files\cuda_7.5.18_win10.exe"; DestDir: "{tmp}"; AfterInstall: CudaInstaller_Win10
Source: "setup_creator_files\cuda_7.5.18_windows.exe"; DestDir: "{tmp}"; AfterInstall: CudaInstaller
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Code]
procedure CudaInstaller_Win10;
var
  ResultCode: Integer;
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  if Version.Major = 10 then begin
    if not Exec(ExpandConstant('{tmp}\cuda_7.5.18_win10.exe'), '', '', SW_SHOWNORMAL,
      ewWaitUntilTerminated, ResultCode)
    then
      MsgBox('Cuda installer failed to run!' + #13#10 +
        SysErrorMessage(ResultCode), mbError, MB_OK);
    end;
end;

procedure CudaInstaller;
var
  ResultCode: Integer;
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  if Version.Major < 10 then begin
    if not Exec(ExpandConstant('{tmp}\cuda_7.5.18_windows.exe'), '', '', SW_SHOWNORMAL,
      ewWaitUntilTerminated, ResultCode)
    then
      MsgBox('Cuda installer failed to run!' + #13#10 +
        SysErrorMessage(ResultCode), mbError, MB_OK);
    end;
end;

[Icons]
Name: "{commonprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

