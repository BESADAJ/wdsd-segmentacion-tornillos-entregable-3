# Conclusiones y limitaciones

## Hallazgos

1. **SGW es el modelo con mayor F1 medio y mejor rango del benchmark** (0,738; rango 2,96 de 15), pero no es significativamente mejor que los cuatro modelos de dos etapas más fuertes.
   - La diferencia con Mask R-CNN en la misma GPU es +0,015, con IC [−0,003; 0,035].
   - Con 14 grupos, el estudio podía detectar diferencias de unos 0,04: la ventaja real, si existe, es menor que eso.
2. **El mecanismo actúa donde la teoría lo predice:** contando todos los tornillos, el recall del tercil pequeño sube de 0,302 a 0,385 (9 de 10 semillas). La ganancia viene de la cabeza RoI y desaparece sin el piso.
3. **Esa ganancia se concentra en una escena**, que tiene el 38 % de los tornillos pequeños de test. La prueba preregistrada por grupo de escena no la confirma (+0,012).
   - La conclusión honesta: SGW ayuda en las escenas con muchos tornillos diminutos. No está demostrado que ayude en general.
4. **SGW cambia recall entre tamaños** (−0,019 en el tercil grande) y ajusta más el entrenamiento (brecha train93 − val de 0,098 frente a 0,059), sin peor generalización entre val y test.
5. **Robustez:**
   - La ventaja de SGW no crece al reducir la escala (la predicción 4 no se cumple): al 50 %, los dos modelos colapsan.
   - SGW es menos robusto que Mask R-CNN al desenfoque (−0,035 a −0,057) y al ruido fuerte (−0,034). Es el principal costo observado del mecanismo.
6. **Generalización:** en BBBC039, NuInsSeg y MinneApple, SGW queda empatado con Mask R-CNN (diferencias de −0,006 a +0,002). Solo es mejor en media en NuInsSeg y no es significativamente peor en ninguno. El criterio de consistencia (2 de 3) no se cumple (capítulo 8 §2).
7. **Teoría:** la similitud SGW tiene una sensibilidad a errores de posición acotada uniformemente en el tamaño (teorema 1), mientras que la del IoU crece como 2/a (proposición 2). Se verificó numéricamente y se cumplió en el piloto de asignación.

## Veredicto sobre R1, R2 y R3

| Condición | Veredicto | Evidencia |
|---|---|---|
| **R1:** modelo nuevo para la aplicación, con fuente citada y diferencia de mecanismo | **Sí** | Capítulo 4: NWD transferido con normalización por objeto y piso, calibración analítica y aplicación en RPN y RoI de segmentación de instancias; ningún trabajo afín lo hace |
| **R2:** mecanismo propio (no un *import*) y propiedad demostrada | **Sí** | `src/original_model.py` desde cero; teorema 1 y proposición 2 con verificación numérica y 17 pruebas |
| **R3:** supera de forma significativa a todo el Top 10 y a los baselines fuertes | **No** | Capítulo 7: equivalente dentro de δ frente a Mask R-CNN, HTC, Cascade y MS R-CNN |

**Tope de calificación:** 4,0, como se declaró antes de entrenar.

## Amenazas a la validez

- **Interna:**
  - el diseño de SGW y la búsqueda de κ usaron la misma partición de validación que el E2 (no hubo una partición de desarrollo aparte para la búsqueda);
  - el resto del Top 10 se corrió en otra GPU y con 5 semillas;
  - el análisis de potencia no se registró a priori.
- **De constructo:**
  - el recall por tercil depende de cortes fijados en val;
  - el F1 por grupo da el mismo peso a grupos con 1 y con 59 tornillos pequeños, y eso decidió H-M.
- **Externa:**
  - un solo conjunto de tornillos;
  - los conjuntos adicionales tienen objetos más grandes y una estructura de grupos pobre (capítulo 8), y en ellos SGW no mantiene su ventaja.
- **Estadística:**
  - 14 grupos limitan la potencia;
  - las diferencias por semilla no son normales (Shapiro p = 0,008), así que el t es orientativo.

## Riesgos y uso responsable

- Un tornillo no detectado puede romper piezas o herir a un operador si el robot fuerza el desensamblado. El modelo pierde cerca de 6 de cada 10 tornillos diminutos y debe usarse con verificación.
- El dataset no declara licencia: no se redistribuyen sus imágenes.

## Trabajo futuro

- Combinar SGW con anclas aún más pequeñas o con una cabeza en P1, porque el piso c = 4 px está atado al paso de P2.
- Asignación por ranking (top-k) con SGW, como NWD-RKA.
- Más escenas con tornillos diminutos, para que la prueba por grupo tenga potencia en el tercil pequeño.

## Declaración de uso de herramientas de IA

El equipo usó un asistente de IA (Claude) para programar, depurar, redactar y revisar la literatura. Todo el código y las cifras se verificaron ejecutándolos. Las decisiones de diseño y de protocolo quedaron fechadas en git antes de ver los resultados confirmatorios (`experiments/procedencia.md`).
