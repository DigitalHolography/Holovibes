; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "Holovibes"
#define MyAppVersion "7.1.1"
#define MyAppPublisher "Holovibes"
#define MyAppURL "http://www.holovibes.com/"
#define MyAppExeName "Holovibes.exe"

#define QtPath "C:\Qt\Qt5.9.0\5.9\msvc2017_64\bin"
#define QtPlatformPath "C:\Qt\Qt5.9.0\5.9\msvc2017_64\plugins\platforms"
#define CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin"

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
SetupIconFile="x64\Release\Holovibes.ico"

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
;Name: "cuda"; Description: "Nvidia and Cuda drivers"; Types: full

[Files]
Source: "x64\Release\Holovibes.exe"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\atmcd64d.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraAdimec.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraIds.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraIxon.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraPCOEdge.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraPCOPixelfly.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraPike.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraPhotonFocus.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraUtils.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraXiq.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraXib.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\CameraHamamatsu.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\FGCamera.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\GPIB.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\m3apiX64.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\xiapi64.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\SC2_Cam.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\uEye_api_64.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\dcamapi.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\adimec.ini"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\edge.ini"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\ids.ini"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\ixon.ini"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\photonfocus.ini"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\pike.ini"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\pixelfly.ini"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\xiq.ini"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\xib.ini"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\hamamatsu.ini"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\Holovibes.ico"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "x64\Release\holovibes_logo.png"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "Holovibes\shaders\fragment.color.glsl"; DestDir: "{app}\{#MyAppVersion}\shaders";Components: program; Flags: ignoreversion
Source: "Holovibes\shaders\fragment.3d.glsl"; DestDir: "{app}\{#MyAppVersion}\shaders";Components: program; Flags: ignoreversion
Source: "Holovibes\shaders\fragment.tex.glsl"; DestDir: "{app}\{#MyAppVersion}\shaders";Components: program; Flags: ignoreversion
Source: "Holovibes\shaders\vertex.direct.glsl"; DestDir: "{app}\{#MyAppVersion}\shaders";Components: program; Flags: ignoreversion
Source: "Holovibes\shaders\vertex.3d.glsl"; DestDir: "{app}\{#MyAppVersion}\shaders";Components: program; Flags: ignoreversion
Source: "Holovibes\shaders\vertex.holo.glsl"; DestDir: "{app}\{#MyAppVersion}\shaders";Components: program; Flags: ignoreversion
Source: "Holovibes\shaders\vertex.overlay.glsl"; DestDir: "{app}\{#MyAppVersion}\shaders";Components: program; Flags: ignoreversion
Source: "InstructionFiles\BatchInstructions.txt"; DestDir: "{app}\{#MyAppVersion}\InstructionFiles";Components: program; Flags: ignoreversion
Source: "InstructionFiles\TriggerWithBurst.txt"; DestDir: "{app}\{#MyAppVersion}\InstructionFiles";Components: program; Flags: ignoreversion
Source: "InstructionFiles\TriggerWithSweep.txt"; DestDir: "{app}\{#MyAppVersion}\InstructionFiles";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\kernel_edge_detect_3.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\kernel_gaussian_3.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\kernel_gradient_x-_3.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\kernel_gradient_x+_3.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\kernel_laplacien_3.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\kernel_prewitt_sobel_2.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\kernel_prewitt_sobel_edge_3.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\kernel_sharpen_3.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\kernel_sobel_xy_5.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\matrix_gradient_x_5.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\matrix_sobel_x_5.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\matrix_sobel_y_5.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\GaussianKernel_256_256.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\GaussianKernel_128_128.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\GaussianKernel_64_64.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\GaussianKernel_64_8.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\GaussianKernel_32_32.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\GaussianKernel_16_16.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\GaussianKernel_8_8.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion
Source: "ConvolutionKernels\GaussianKernel_4_4.txt"; DestDir: "{app}\{#MyAppVersion}\ConvolutionKernels";Components: program; Flags: ignoreversion


Source: "{#QtPath}\Qt5Core.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Gui.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5OpenGL.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5PrintSupport.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Widgets.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Svg.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPath}\Qt5Charts.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#QtPlatformPath}\*"; DestDir: "{app}\{#MyAppVersion}\platforms";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\cufft64_10.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#CudaPath}\cudart64_101.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "Camera\CameraAdimec\bfml\Adimec-Quartz-2A750-Mono_12bit.bfml"; DestDir: "{app}\{#MyAppVersion}\Camera\CameraAdimec\bfml";Components: program; Flags: ignoreversion
Source: "Camera\CameraAdimec\bfml\Adimec Q2A750m-0.4-0.3 (4x freerun).bfml"; DestDir: "{app}\{#MyAppVersion}\Camera\CameraAdimec\bfml";Components: program; Flags: ignoreversion
Source: "Camera\CameraAdimec\bfml\Adimec Q2A750m-0.2 (4x freerun).bfml"; DestDir: "{app}\{#MyAppVersion}\Camera\CameraAdimec\bfml";Components: program; Flags: ignoreversion
Source: "setup_creator_files\vcredist_2017_x64.exe"; DestDir: "{tmp}";Components: visual; Flags: nocompression ignoreversion; AfterInstall: Visual2017
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[UninstallDelete]
Type: files; Name: "{app}\{#MyAppVersion}\holovibes.ini"
Type: filesandordirs; Name: "{app}\{#MyAppVersion}"

[Code]

procedure Visual2017;
var                                   
  ResultCode: Integer;
begin
    if not Exec(ExpandConstant('{tmp}\vcredist_2017_x64.exe'), '', '', SW_SHOWNORMAL,
      ewWaitUntilTerminated, ResultCode)
    then
      MsgBox('Visual c++ redistributable 2017 failed to run!' + #13#10 +
        SysErrorMessage(ResultCode), mbError, MB_OK);
end;

[Icons]
Name: "{commonprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppVersion}\{#MyAppExeName}"                                                                                                                                                                                                                                                                                                                                                                                                                                                  
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppVersion}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppVersion}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppVersion}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

