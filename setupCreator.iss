; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

; CMake: build/Generator      VisualStudio: x64
#define BuildDir 

#define MyAppName "Holovibes"
#define MyAppVersion 1
#define MyAppPublisher "Digital Holography"
#define MyAppURL "http://www.holovibes.com/"
#define MyAppExeName Holovibes.exe
#define MyLicense "LICENSE"


#define cmakePath C:/Users/Karachayevsk/.conan/data/cmake/3.21.1/_/_/package/01edd76db8e16db9b38c3cca44ec466a9444c388

#define qtPath C:/.conan/fa22b0/1

#define boostPath C:/.conan/f564a1/1

#define glmPath C:/Users/Karachayevsk/.conan/data/glm/0.9.9.8/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9

#define gtestPath C:/Users/Karachayevsk/.conan/data/gtest/1.10.0/_/_/package/875c67f4d8a79bdd002908b75efce119eb82836d

#define nlohmann_jsonPath C:/Users/Karachayevsk/.conan/data/nlohmann_json/3.10.4/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9

#define opencvPath C:/.conan/877c73/1

#define openglPath C:/Users/Karachayevsk/.conan/data/opengl/system/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9

#define pcre2Path C:/Users/Karachayevsk/.conan/data/pcre2/10.37/_/_/package/d5bf9f9853f79bcb782c6a7ad39636782a3da0fd

#define double-conversionPath C:/Users/Karachayevsk/.conan/data/double-conversion/3.1.5/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define harfbuzzPath C:/.conan/eb7c61/1

#define sqlite3Path C:/Users/Karachayevsk/.conan/data/sqlite3/3.36.0/_/_/package/5e2ee2bbbf247701a335de1962053720f3cbabf1

#define libpqPath C:/Users/Karachayevsk/.conan/data/libpq/13.2/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define jasperPath C:/Users/Karachayevsk/.conan/data/jasper/2.0.32/_/_/package/2278242cc765d77e29f54e7b3e73f056c18d9a10

#define openexrPath C:/Users/Karachayevsk/.conan/data/openexr/2.5.7/_/_/package/6acf24cd4adf2df742e006cc0e5f0329e3b6e60b

#define libtiffPath C:/Users/Karachayevsk/.conan/data/libtiff/4.3.0/_/_/package/e2e95cb5797e0ea720de15db3e7a049a3772be97

#define eigenPath C:/Users/Karachayevsk/.conan/data/eigen/3.3.9/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9

#define ffmpegPath C:/Users/Karachayevsk/.conan/data/ffmpeg/4.4/_/_/package/81c9ad7687710b78c9f840db13ed7055b1ea240a

#define quircPath C:/Users/Karachayevsk/.conan/data/quirc/1.1/_/_/package/11b7230897ba30e6da1af1c8b0726ed134f404ee

#define protobufPath C:/.conan/dbe8c7/1

#define adePath C:/Users/Karachayevsk/.conan/data/ade/0.1.1f/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define opensslPath C:/Users/Karachayevsk/.conan/data/openssl/1.1.1l/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define freetypePath C:/Users/Karachayevsk/.conan/data/freetype/2.11.0/_/_/package/c69bad48274e7fc45c887cddc32aa0459f0a81e0

#define glibPath C:/.conan/65c410/1

#define libjpegPath C:/Users/Karachayevsk/.conan/data/libjpeg/9d/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define libdeflatePath C:/Users/Karachayevsk/.conan/data/libdeflate/1.8/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define xz_utilsPath C:/Users/Karachayevsk/.conan/data/xz_utils/5.2.5/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define jbigPath C:/Users/Karachayevsk/.conan/data/jbig/20160605/_/_/package/a05a9ef21a91686b7138c926ec52aaeb70439310

#define zstdPath C:/Users/Karachayevsk/.conan/data/zstd/1.5.0/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define libwebpPath C:/Users/Karachayevsk/.conan/data/libwebp/1.2.0/_/_/package/092f77d3b4b4678d81fbffc1fccc9642b870175e

#define openjpegPath C:/Users/Karachayevsk/.conan/data/openjpeg/2.4.0/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define openh264Path C:/Users/Karachayevsk/.conan/data/openh264/2.1.1/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define vorbisPath C:/Users/Karachayevsk/.conan/data/vorbis/1.3.7/_/_/package/b8a7c0e3d5f7f2dd0a4cf862cb60ff9f1b73be47

#define opusPath C:/Users/Karachayevsk/.conan/data/opus/1.3.1/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define libx264Path C:/Users/Karachayevsk/.conan/data/libx264/20191217/_/_/package/e03f63a768ca59a76d565ff28923e7805edec16d

#define libx265Path C:/Users/Karachayevsk/.conan/data/libx265/3.4/_/_/package/1287c7aad58fe35bb1a62d9fcc674752df4a764a

#define libvpxPath C:/Users/Karachayevsk/.conan/data/libvpx/1.10.0/_/_/package/0a69d6ddb06b8782b01aad521e2a0c2edf39870a

#define libmp3lamePath C:/Users/Karachayevsk/.conan/data/libmp3lame/3.100/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define libfdk_aacPath C:/Users/Karachayevsk/.conan/data/libfdk_aac/2.0.2/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define libpngPath C:/Users/Karachayevsk/.conan/data/libpng/1.6.37/_/_/package/8b1ef0ec9599374db4689199730c00a0d5f4de36

#define brotliPath C:/Users/Karachayevsk/.conan/data/brotli/1.0.9/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define libffiPath C:/Users/Karachayevsk/.conan/data/libffi/3.4.2/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define pcrePath C:/Users/Karachayevsk/.conan/data/pcre/8.45/_/_/package/7a162084f5550c625a8950e9b1175d1474d2fbab

#define libelfPath C:/Users/Karachayevsk/.conan/data/libelf/0.8.13/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define libgettextPath C:/Users/Karachayevsk/.conan/data/libgettext/0.20.1/_/_/package/e0c22822cdf05b624135e1696ae5cb784a23aeb3

#define oggPath C:/Users/Karachayevsk/.conan/data/ogg/1.3.4/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define zlibPath C:/Users/Karachayevsk/.conan/data/zlib/1.2.11/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714

#define bzip2Path C:/Users/Karachayevsk/.conan/data/bzip2/1.0.8/_/_/package/589a23dff5fdb23a7fb851223eb766480ead0a9a

#define libiconvPath C:/Users/Karachayevsk/.conan/data/libiconv/1.16/_/_/package/d057732059ea44a47760900cb5e4855d2bea8714




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
LicenseFile={#MyLicense}
OutputBaseFilename=holovibes_setup_{#MyAppVersion}
Compression=lzma
SolidCompression=yes
DiskSpanning=no
UninstallDisplayName=Holovibes
UninstallDisplayIcon={app}\{#MyAppVersion}\Holovibes.exe
UninstallFilesDir={app}\{#MyAppVersion}
SetupIconFile="{#BuildDir}\Holovibes.ico"
ChangesAssociations=yes
UsePreviousAppDir=no

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

Source: "{#qtPath}\bin\Qt6Core.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#qtPath}\bin\Qt6Gui.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#qtPath}\bin\Qt6OpenGL.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#qtPath}\bin\Qt6OpenGLWidgets.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#qtPath}\bin\Qt6PrintSupport.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#qtPath}\bin\Qt6Widgets.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#qtPath}\bin\Qt6Svg.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#qtPath}\bin\Qt6Charts.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#qtPath}\bin\res\archdatadir\plugins\platforms\*"; DestDir: "{app}\{#MyAppVersion}\platforms";Components: program; Flags: ignoreversion
Source: "{#cudaPath}\bin\cufft64_10.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#cudaPath}\bin\cublas64_11.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#cudaPath}\bin\cublasLt64_11.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#cudaPath}\bin\cusolver64_11.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#cudaPath}\bin\cudart64_110.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#opencvPath}\bin\opencv_videoio_ffmpeg450_64.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion
Source: "{#opencvPath}\bin\opencv_world450.dll"; DestDir: "{app}\{#MyAppVersion}";Components: program; Flags: ignoreversion

Source: "Camera\configs\*.ini"; DestDir: "{userappdata}\{#MyAppName}\{#MyAppVersion}\cameras_config"; Components: program; Flags: ignoreversion recursesubdirs

Source: "resources\setup_creator_files\vcredist_2019_x64.exe"; DestDir: "{tmp}"; Components: visual; Flags: nocompression ignoreversion;

[Dirs]
Name: "{userappdata}\{#MyAppName}\{#MyAppVersion}"
Name: "{userappdata}\{#MyAppName}\{#MyAppVersion}\cameras_config"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\{#MyAppVersion}"
Type: filesandordirs; Name: "{userappdata}\{#MyAppName}\{#MyAppVersion}"
Type: dirifempty; Name: "{userappdata}\{#MyAppName}"

[Run]
Filename: "{tmp}\vcredist_2019_x64.exe"; Parameters: "/install /passive /norestart"; Components: visual; Flags: waituntilterminated

[Icons]
Name: "{commonprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppVersion}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppVersion}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
Root: HKCR; Subkey: ".holo"; ValueData: "{#MyAppName}"; Flags: uninsdeletevalue; ValueType: string; ValueName: ""
Root: HKCR; Subkey: "{#MyAppName}"; ValueData: "{#MyAppName}"; Flags: uninsdeletekey; ValueType: string; ValueName: ""
Root: HKCR; Subkey: "{#MyAppName}\DefaultIcon"; ValueData: "{app}\{#MyAppVersion}\{#MyAppExeName},0"; ValueType: string; ValueName: ""
Root: HKCR; Subkey: "{#MyAppName}\shell\open\command"; ValueData: """{app}\{#MyAppVersion}\{#MyAppExeName}"" ""-i"" ""%1"""; ValueType: string;  ValueName: ""

