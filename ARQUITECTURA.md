# ARQUITECTURA.md

## Visión

NeuroGraph es un RAG (Retrieval-Augmented Generation) personal, local y agnóstico respecto a la fuente de datos. Su objetivo es permitir indexar y recuperar información desde múltiples fuentes externas (archivos locales, notas, servicios de terceros) sin acoplar el sistema a ninguna de ellas, manteniendo un presupuesto de infraestructura de $0 mediante un monolito modular.

## Dominios

El sistema se divide inicialmente en cuatro dominios estrictos:

```text
parsing
    Extracción de información desde fuentes externas.
    Produce datos brutos.

transform
    Normalización y transformación de datos brutos
    hacia los contratos internos.

models
    Define los contratos de datos compartidos entre dominios.

retrieval
    Fragmentación, indexación y recuperación de información.
```

## Regla fundamental de responsabilidades

Cada dominio debe hacer exclusivamente su trabajo. Ningún dominio debe asumir responsabilidades pertenecientes a otro dominio.

En particular:

```text
parsing ≠ transform
transform ≠ retrieval
retrieval ≠ parsing
models ≠ lógica de negocio
```

`parsing` extrae contenido crudo de una fuente y lo entrega como `RawData`, sin normalizar ni interpretar su significado.

`transform` es responsable de convertir `RawData` en `Document`, aplicando normalización y reglas de negocio. `transform` no se implementa en esta fase.

`retrieval` es responsable de fragmentar `Document` en `Chunk`, indexar y recuperar información. `retrieval` no se implementa en esta fase.

`models` únicamente define los contratos de datos (`RawData`, `Document`, `Chunk`) que permiten a los demás dominios comunicarse entre sí. `models` no contiene lógica de negocio.

## YAGNI

No se introducen abstracciones, interfaces, factories, adapters, repositories, dependency injection ni otros patrones arquitectónicos hasta que exista una necesidad concreta que los justifique.

La simplicidad es una restricción arquitectónica, no una omisión temporal.

## Pydantic

Pydantic se utiliza exclusivamente para definir y validar contratos en las fronteras de los dominios (`models`). No se utiliza como estructura interna genérica de toda la aplicación.
