; Jag Studio installer (Inno Setup 6). Built by .github/workflows/windows-installer.yml:
;   iscc /DMyAppVersion=0.1.N packaging\windows\installer.iss
; Installs for the current user only (no administrator rights needed).
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif

[Setup]
AppId={{6F1E2B7C-4D3A-4C55-9C1E-8A2B7D5E3F10}
AppName=Jag Studio
AppVersion={#MyAppVersion}
AppPublisher=SPB Textile
DefaultDirName={localappdata}\Programs\Jag Studio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=Jag Studio Setup
SetupIconFile=jagstudio.ico
UninstallDisplayIcon={app}\Jag Studio.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\..\dist\Jag Studio\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\Jag Studio"; Filename: "{app}\Jag Studio.exe"
Name: "{autodesktop}\Jag Studio"; Filename: "{app}\Jag Studio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Jag Studio.exe"; Description: "Start Jag Studio"; Flags: nowait postinstall skipifsilent
