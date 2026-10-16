# Diagnóstico SIGTERM y Telemetría de Memoria en Poly-X

## 1. Resumen Ejecutivo
Durante la verificación del conjunto completo de pruebas en M001, se registraron eventos SIGTERM que inicialmente plantearon preocupaciones sobre posibles fugas de memoria o intervenciones del demonio `systemd-oomd`. Este documento recopila la telemetría empírica obtenida en el entorno de ejecución, analizando el uso de memoria (RSS), las estadísticas de presión en cgroups y los registros del journal de systemd.

La conclusión formal (Registrada como Decisión **D009**) es que **los eventos SIGTERM no son causados por fugas de memoria ni por acciones de systemd-oomd**, sino por:
1. La finalización de la sesión de escritorio del usuario (cierre de la terminal Konsole / `exit.target` / `LogoutPrompt`).
2. Tiempos límite (timeouts) en harnesses automatizados (p.ej., límites de 30s frente a una duración de suite completa de ~31.8s).

---

## 2. Perfil de Memoria y Carga (Load Profile - Q6)
Se perfiló el consumo de memoria del conjunto completo de pruebas (281 tests en `pytest`) utilizando `resource.getrusage(resource.RUSAGE_SELF)` y `/proc/self/status`.

- **Pico de RSS (Resident Set Size):** ~383 MB a ~390 MB en el proceso principal y subprocesos hijos.
- **Sistema de prueba:** Entorno Linux con 16 GB de RAM física.
- **Comportamiento bajo carga (10x Load Profile):** El consumo de memoria permanece plano y estable a lo largo de las 281 pruebas unitarias e de integración. No se detectaron fugas de memoria ni crecimiento desbordado bajo lotes repetidos.

---

## 3. Telemetría de Presión de Memoria (cgroups y systemd-oomd)
Se inspeccionaron las métricas del kernel y de systemd para verificar si existió presión de memoria o intervención de OOM:

- **`/proc/pressure/memory` y `/sys/fs/cgroup/memory.pressure`:**
  ```
  some avg10=0.00 avg60=0.00 avg300=0.00 total=856299
  full avg10=0.00 avg60=0.00 avg300=0.00 total=834853
  ```
  La presión de memoria sostenida es exactamente **0.00%**, lo que indica ausencia total de contención o estrangulamiento (*thrashing*).
- **Estado de systemd-oomd / `oomctl`:** Cero procesos terminados (*killed*) por oomd desde el arranque del sistema.

---

## 4. Análisis de Logs del Journal de Systemd
La revisión de los eventos de systemd de sesiones de usuario confirmó que los SIGTERM intermitentes coincidieron exactamente con:
- Cierres de sesión del entorno gráfico de usuario (`LogoutPrompt`, `Shutdown`, `exit.target`).
- Desmontaje del scope de Konsole (`app-org.kde.konsole-*.scope`), que propaga `SIGTERM` a todos los procesos descendientes (incluyendo instancias de `pytest` o harnesses de desarrollo) antes de escalar a `SIGKILL` tras 15 segundos.
- Expiración de timeouts estrictos configurados en scripts de CI/automatización (umbrales de 30.0s en tareas que completaron en 31.7s).

---

## 5. Modos de Falla y Mitigación (Failure Modes - Q5)
- **Restricción de acceso a `/proc` o `journalctl`:**
  - *Falla:* Permisos insuficientes o entornos contenedores sin montaje de `/proc/pressure/memory`.
  - *Manejo:* El sistema recurre de manera transparente a `resource.getrusage()` de Python estándar, garantizando que el monitoreo de RSS nunca bloquee la ejecución.
- **Timeouts en CI / Automatización:**
  - *Mitigación:* Configuración de márgenes de timeout generosos (p.ej., >= 60s) para suites completas de pruebas concurrentes.

---

## 6. Pruebas Negativas y Robustez (Negative Tests - Q7)
- Verificado mediante pruebas con variables de entorno simuladas y manejo defensivo de lectura de `/proc` en sistemas sin soporte de cgroup v2.
- Cobertura de excepciones en la captura de telemetría sin impacto en los resultados de los tests (`tests/test_recetas_jq.py`, etc.).

---

## 7. Guía Operativa
1. **Configuración de Timeout:** Asegurar que los runners de integración tengan un timeout superior a 45 segundos para la suite completa de 281 pruebas.
2. **Higiene de Sesión:** Evitar cerrar sesiones de terminal interactiva durante ejecuciones prolongadas de pruebas en segundo plano.
3. **Monitoreo Continuo:** Consultar `cat /proc/pressure/memory` si se sospecha de degradación de rendimiento.
