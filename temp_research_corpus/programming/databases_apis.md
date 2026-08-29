---
title: "Bases de Datos y APIs REST en Python"
tags: [programacion, python, apis, bases-de-datos]
---

# Bases de Datos y APIs REST

Las aplicaciones backend en Python suelen combinar una base de datos
persistente con una API que expone los datos a clientes externos.

## Tipos de bases de datos

Existen dos grandes familias utilizadas comúnmente:

- **Bases de datos relacionales** (PostgreSQL, MySQL): organizan los datos en tablas con esquemas fijos y soportan consultas SQL complejas.
- **Bases de datos NoSQL** (MongoDB, Redis): ofrecen esquemas flexibles y suelen priorizar velocidad o escalabilidad horizontal sobre consistencia estricta.

## Diseño de APIs REST

Una API REST expone recursos mediante URLs y verbos HTTP (GET, POST, PUT,
DELETE). Los principios de diseño incluyen usar sustantivos en las rutas,
códigos de estado HTTP coherentes y versionado explícito de la API.

En Python, librerías como SQLAlchemy actúan como ORM (mapeo
objeto-relacional), permitiendo manipular filas de la base de datos como si
fueran objetos, sin escribir SQL manualmente en cada consulta.
