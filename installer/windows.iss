#define MyAppName "Momo-LM"
#define MyAppVersion GetEnv("MOMO_VERSION")
#define MyAppPublisher "Momo-LM contributors"
#define MyAppURL "https://github.com/YanagiKH/Momo-LM"
#define MyAppExeName "Momo-LM.exe"

[Setup]
AppId={{B870535B-C81B-4730-A942-7669BB626A35}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\release
OutputBaseFilename=Momo-LM-Setup-Windows-x64
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesetraditional"; MessagesFile: "{#SourcePath}\languages\ChineseTraditional.isl"
Name: "japanese"; MessagesFile: "{#SourcePath}\languages\Japanese.isl"

[Files]
Source: "..\dist\Momo-LM\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Momo-LM"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Momo-LM"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Momo-LM"; Flags: nowait postinstall skipifsilent
