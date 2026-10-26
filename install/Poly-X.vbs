' Poly-X — lanzador sin consola.
'
' Abre el Launcher directamente, sin ventana negra. Resuelve su propia ubicacion,
' asi que funciona desde cualquier carpeta: basta descargar el repo y hacer doble
' clic, sin editar rutas.
'
' Usa pythonw.exe (el interprete sin consola) en vez de python.exe.

Option Explicit

Dim fso, shell, raiz, pythonw, comando

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

raiz = fso.GetParentFolderName(WScript.ScriptFullName)
raiz = fso.GetParentFolderName(raiz)
pythonw = raiz & "\.venv\Scripts\pythonw.exe"

If Not fso.FileExists(pythonw) Then
    MsgBox "No se encuentra el entorno virtual." & vbCrLf & vbCrLf & _
           "Buscado en:" & vbCrLf & pythonw & vbCrLf & vbCrLf & _
           "Ejecuta primero SETUP.bat en esta misma carpeta.", _
           vbCritical, "Poly-X"
    WScript.Quit 1
End If

If Not fso.FolderExists(raiz & "\src\polyx") Then
    MsgBox "No se encuentra la carpeta 'polyx' junto a este archivo." & vbCrLf & _
           vbCrLf & "Deja Poly-X.vbs en install\, junto a SETUP.bat.", _
           vbCritical, "Poly-X"
    WScript.Quit 1
End If

' El directorio de trabajo debe ser la raiz: 'python -m polyx.launcher' importa
' el paquete desde ahi. Con el layout src/, src/ debe estar en PYTHONPATH.
shell.Environment("Process")("PYTHONPATH") = raiz & "\src;" & shell.Environment("Process")("PYTHONPATH")
shell.CurrentDirectory = raiz

' Estilo de ventana 0 = oculta; False = no esperar a que termine.
comando = """" & pythonw & """ -m polyx.launcher"
shell.Run comando, 0, False
