---
title: "Testing Automatizado y Despliegue Continuo"
tags: [programacion, testing, ci-cd, docker]
---

# Testing Automatizado y Despliegue Continuo

Garantizar la calidad de un servicio backend requiere pruebas automatizadas
ejecutadas de forma sistemática antes de cada despliegue.

## Tipos de pruebas

- **Pruebas unitarias**: verifican una función o clase de forma aislada.
- **Pruebas de integración**: verifican que varios componentes funcionen correctamente juntos.
- **Pruebas end-to-end**: simulan el comportamiento real de un usuario sobre el sistema completo.

## Integración y despliegue continuo (CI/CD)

Un pipeline de CI/CD ejecuta automáticamente las pruebas cada vez que se sube
código nuevo, y si todas pasan, despliega la aplicación a producción sin
intervención manual. Herramientas como GitHub Actions o GitLab CI son
comunes para orquestar estos pipelines.

El empaquetado en contenedores Docker permite que una aplicación Python se
ejecute de forma idéntica en desarrollo, pruebas y producción, eliminando el
clásico problema de "en mi máquina funciona".
