; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "Holovibes"
#define MyAppVersion "4.0.0"
#define MyAppPublisher "CNRS"
#define MyAppURL "http://www.holovibes.com/"
#define MyAppExeName "Holovibes.exe"
;define for Qt ,Qwt path and Cuda
#define QtPath "C:\Qt\Qt5.7.1\5.7\msvc2013_64\bin"
#define QtPlatformPath "C:\Qt\Qt5.7.1\5.7\msvc2013_64\plugins\platforms"
#define QwtPath "C:\Qt\qwt-6.1.3\lib"
#define CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
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
SetupIconFile="x64\Release\icon1.ico"

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
Name: "cuda"; Description: "Nvidia and Cuda drivers"; Types: full

[Files]
Source: "x64\Release\Holovibes.exe"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\atmcd64d.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraAdimec.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraIds.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraIxon.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraPCOEdge.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraPCOPixelfly.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraPike.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraUtils.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraXiq.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\FGCamera.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\GPIB.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\m3apiX64.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\SC2_Cam.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\uEye_api_64.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\edge.ini"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\ids.ini"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\ixon.ini"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\pike.ini"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\pixelfly.ini"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\xiq.ini"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\icon1.ico"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "x64\Release\holovibes_logo.png"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "{#QwtPath}\qwt.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "{#QwtPath}\qwtd.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Core.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Gui.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5OpenGL.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5PrintSupport.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Widgets.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Svg.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "{#QtPlatformPath}\*"; DestDir: "{app}\platforms";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\cufft64_80.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\cudart64_80.dll"; DestDir: "{app}";Components: program; Flags: ignoreversion
Source: "setup_creator_files\vcredist_x64.exe"; DestDir: "{tmp}";Components: visual; Flags: nocompression ignoreversion; AfterInstall: Visual
Source: "setup_creator_files\cuda_8.0.44_win10.exe"; DestDir: "{tmp}"; Components: cuda; Flags: nocompression ignoreversion; Check : CheckWin10; AfterInstall: CudaInstaller_Win10 
Source: "setup_creator_files\cuda_8.0.44_windows.exe"; DestDir: "{tmp}"; Components: cuda; Flags: nocompression ignoreversion; Check : CheckWinOther; AfterInstall: CudaInstaller
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[UninstallDelete]
Type: files; Name: "{app}\holovibes.ini"
Type: filesandordirs; Name: "{app}"

[Code]
function CheckWin10 : Boolean;
var
  ResultCode: Integer;
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  if Version.Major >= 10 then begin
      Result := True;
  end else begin
      Result := False;
  end;
end;

function CheckWinOther : Boolean;
var
  ResultCode: Integer;
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  if Version.Major < 10 then begin
      Result := True;
  end else begin
      Result := False;
  end;
end;

procedure Visual;
var
  ResultCode: Integer;
begin
    if not Exec(ExpandConstant('{tmp}\vcredist_x64.exe'), '', '', SW_SHOWNORMAL,
      ewWaitUntilTerminated, ResultCode)
    then
      MsgBox('Visual c++ redistributable 2013 failed to run!' + #13#10 +
        SysErrorMessage(ResultCode), mbError, MB_OK);
end;

procedure CudaInstaller_Win10;
var
  ResultCode: Integer;
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  if Version.Major >= 10 then begin
    if not Exec(ExpandConstant('{tmp}\cuda_8.0.44_win10.exe'), '', '', SW_SHOWNORMAL,
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
    if not Exec(ExpandConstant('{tmp}\cuda_8.0.44_windows.exe'), '', '', SW_SHOWNORMAL,
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

