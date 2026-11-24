# Modelo original: diseño, hipótesis y teoría

**Estado:** borrador del 2026-11-23, antes de cualquier entrenamiento del modelo original. Las cifras del E2 que se citan son las del repositorio del E2 (commit `732845a`).

## 1. Limitación que se ataca (D1)

**Los modelos de la familia no detectan los tornillos pequeños, y la ceguera es compartida.** Evidencia del E2 (RQ3 y errores compartidos, `results/e2/analisis_extra/` y `analisis_extra2/` del repositorio del E2):

| | tercil pequeño (área < 75 px) | mediano (75–157 px) | grande (> 157 px) |
|---|---|---|---|
| Recall de Mask R-CNN (paper guía del E1, mejor modelo del E2) | 0,31 | 0,69 | 0,79 |
| Recall de HTC / Cascade / MS R-CNN | 0,32 / 0,28 / 0,30 | 0,70 / 0,69 / 0,71 | 0,80 / 0,78 / 0,75 |
| Tornillos de test en el tercil | 155 | 117 | 117 |

- 100 de los 389 tornillos de test no los encuentra **ninguno** de los 13 modelos; los tres mejores comparten el 74 % de sus fallos.
- La correlación media de errores entre modelos es 0,82.
- Es decir, la limitación no es de una arquitectura sino de algo común a toda la familia. Lo común a los detectores de dos etapas y a la mayoría de los de una etapa es la **asignación de etiquetas por IoU**.

Pendiente antes de congelar: el piloto de asignación (sección 6) sobre la partición de desarrollo, que mide cuántas anclas positivas recibe cada tornillo según su tamaño.

## 2. Mecanismo

### 2.1 Fuente y por qué se transfiere (D2)

**Fuente:** NWD, *Normalized Gaussian Wasserstein Distance* (Wang, Xu, Yang y Yu, 2021, arXiv:2110.13389; versión de revista con el benchmark AI-TOD: Xu et al., *ISPRS J. Photogramm. Remote Sens.*, 2022). Viene de la detección de objetos diminutos en imágenes aéreas. Modela cada caja como una gaussiana 2D y reemplaza el IoU por

$$\mathrm{NWD}(a,b) = \exp\!\left(-\frac{W_2(\mathcal N_a,\mathcal N_b)}{C}\right),$$

con $C$ una constante global (el tamaño medio de los objetos).

**Propiedad estructural compartida:** el tornillo mediano de WDSD mide 10,6 px de diámetro equivalente y el 98,8 % es *small* según COCO, igual que los objetos de AI-TOD. Con objetos de 8–12 px, un error de 2–3 px de la rejilla de anclas, cuyo paso es 4 px en P2, basta para que el IoU caiga por debajo del umbral, y el tornillo se queda sin ejemplos positivos.

**Por qué no basta con NWD tal cual:** en WDSD el lado de los tornillos va de unos 3 a unos 60 px. Con una $C$ fija, la tolerancia es la misma en píxeles para todos. Resulta demasiado laxa para los tornillos diminutos frente a su propio tamaño y demasiado estricta para los grandes. Además, los umbrales de Mask R-CNN (0,7/0,3 en RPN, 0,5 en RoI) fueron pensados para el IoU y no tienen una interpretación clara en NWD.

### 2.2 Modificación algorítmica (lo propio)

Sea $g=(x_1,y_1,x_2,y_2)$ la caja de la verdad y $b$ una caja candidata (ancla o propuesta). Se definen:

$$p(b) = \left(\tfrac{x_1+x_2}{2},\ \tfrac{y_1+y_2}{2},\ \tfrac{x_2-x_1}{2},\ \tfrac{y_2-y_1}{2}\right), \qquad W(g,b) = \lVert p(g)-p(b)\rVert_2 \tag{1}$$

$W$ es la distancia de Wasserstein-2 entre $\mathcal N\big((c_x,c_y),\mathrm{diag}(w^2/4,h^2/4)\big)$ de cada caja (Wang et al., 2021, ec. 5).

$$D(g) = \kappa\left(\sqrt{w_g h_g} + c\right), \qquad s(g,b) = \exp\!\left(-\frac{W(g,b)}{D(g)}\right) \tag{2}$$

Las tres decisiones de diseño:

1. **Escala por objeto en lugar de la constante global $C$.** $D$ depende de la verdad, no del conjunto. $s$ no es simétrica, y no hace falta que lo sea: en la asignación la escala debe fijarla el objeto que se quiere encontrar.
2. **Piso $c$ = 4 px**, el paso de la rejilla de anclas en P2. Se fija a priori y no se ajusta. Impide que la tolerancia se anule en los objetos diminutos y da la cota de Lipschitz del teorema 1.
3. **Calibración de $\kappa$ con el IoU.** En el límite de objetos grandes ($c$ despreciable), se pide que $s = 0{,}5$ donde $\mathrm{IoU}=0{,}5$:
   - traslación de $w/3$: $\kappa = 1/(3\ln 2) = 0{,}481$;
   - cambio de escala de $\sqrt 2$: $\kappa = (\sqrt2-1)/(\sqrt2\ln 2) = 0{,}423$.

   Se toma $\kappa_0 = 0{,}45$, y los umbrales de Mask R-CNN se dejan como están. La búsqueda es $\kappa\in\kappa_0\times\{0{,}5;\,1;\,2\}$, con $T=3$ configuraciones, el mismo presupuesto que la tasa de aprendizaje de cada modelo del E2.

$s$ reemplaza al IoU **en las dos asignaciones de Mask R-CNN**:
- anclas → verdad, en RPN;
- propuestas → verdad, en la cabeza RoI. De esta asignación dependen también los ejemplos que entrenan la rama de máscara.

Todo lo demás (backbone, anclas de 8–128 px, pérdidas, presupuesto) es idéntico al Mask R-CNN del E1/E2. Código: `src/original_model.py`.

### 2.3 Pseudocódigo

```
Algoritmo: asignación SGW (una imagen)
Entrada:  cajas de verdad G (n×4), candidatas B (m×4), kappa > 0, c > 0, umbrales (t_alto, t_bajo)
Salida:   para cada candidata, índice de la verdad asignada, o FONDO, o IGNORAR
1. P_G <- p(G); P_B <- p(B)                               # ec. (1)
2. W[i,j] <- ||P_G[i] - P_B[j]||_2                        # matriz n×m
3. D[i] <- kappa * (sqrt(w_i h_i) + c)                    # ec. (2)
4. S[i,j] <- exp(-W[i,j] / D[i])
5. para cada candidata j: i* <- argmax_i S[i,j]; q <- S[i*,j]
       si q >= t_alto: asignar i*;  si q < t_bajo: FONDO;  si no: IGNORAR
6. (solo RPN) para cada verdad i: las candidatas con S[i,j] = max_j S[i,j] se asignan a i   # baja calidad
Criterio de parada: no aplica (asignación de un paso, sin iteración).
Complejidad: O(n m) en tiempo y memoria, igual que el IoU.
```

## 3. Teoría

**Supuestos.**
- (S1) Cajas alineadas con los ejes, con $w,h>0$.
- (S2) La caja se representa por la gaussiana de la sección 2.2. Es un modelo de la caja, no del objeto: los tornillos son aproximadamente circulares, así que la gaussiana isotrópica es razonable, pero las máscaras no son elipses.
- (S3) $\kappa>0$ y $c>0$.

**Teorema 1 (estabilidad de la asignación SGW).** Bajo (S1)–(S3), para toda verdad $g$ y candidatas $b,b'$:

- **(a) Rango e identidad.** $0 < s(g,b) \le 1$, y $s(g,b)=1 \iff b=g$.
- **(b) Cota de Lipschitz uniforme.**
$$|s(g,b)-s(g,b')| \;\le\; \frac{\lVert p(b)-p(b')\rVert_2}{D(g)} \;\le\; \frac{\lVert b-b'\rVert_2}{\sqrt2\,\kappa\,c}.$$
La constante no depende del tamaño de $g$.
- **(c) Sensibilidad no nula.** Si $b\ne g$, $s$ es diferenciable en $p(b)$ y $\lVert\nabla_{p(b)} s\rVert = s/D(g) > 0$.
- **(d) Escala.** Sea $\rho(g)=\sqrt{A_g}/(\sqrt{A_g}+c)\in(0,1)$, con $A_g=w_gh_g$, y sea $s_0$ la similitud con $c=0$. Entonces $s(g,b)=s_0(g,b)^{\rho(g)}$, y para $\lambda>0$, $s_c(\lambda g,\lambda b)=s_{c/\lambda}(g,b)$. En consecuencia:
  - $s_0$ es invariante a la escala;
  - $s\to s_0$ cuando $A_g\to\infty$;
  - cuando $A_g\to 0$, $s\to\exp(-W/(\kappa c))$, que es NWD con $C=\kappa c$.

**Proposición 2 (el IoU no tiene cota uniforme).** Sea $g$ un cuadrado de lado $a$ y $b$ su traslación en $t\in[0,a)$ píxeles a lo largo de un eje. Entonces $\mathrm{IoU}=(a-t)/(a+t)$ y $\left|\partial\,\mathrm{IoU}/\partial t\right|_{t=0} = 2/a$, que no está acotado cuando $a\to0$. Además, $\mathrm{IoU}\equiv0$ en el abierto de candidatas disjuntas de $g$, donde su gradiente es nulo.

**Corolario 3 (margen de estabilidad).** Si una candidata tiene $s(g,b)\ge \tau+m$ con $m>0$, sigue siendo positiva ($s\ge\tau$) ante cualquier perturbación con $\lVert b-b'\rVert_2 < \sqrt2\,\kappa\,c\,m$. Con $\kappa_0=0{,}45$ y $c=4$: $2{,}55\,m$ px, para cualquier tamaño de tornillo.

En cambio, con el IoU, en un tornillo de 8 px un desplazamiento de 1 px resta unos 0,22 de IoU.

### Demostraciones

**(a)** $W\ge0$ y $D>0$, así que $s=e^{-W/D}\in(0,1]$. $s=1 \iff W=0 \iff p(b)=p(g)$, y $p$ es una biyección lineal de $\mathbb R^4$ (ver (b)), así que $p(b)=p(g)\iff b=g$. ∎

**(b)** Sea $\varphi(x)=e^{-x/D}$ en $[0,\infty)$. Como $|\varphi'(x)|=e^{-x/D}/D\le 1/D$, por el teorema del valor medio $|\varphi(x)-\varphi(y)|\le|x-y|/D$. Por la desigualdad triangular inversa, $|W(g,b)-W(g,b')| = \big|\lVert p(g)-p(b)\rVert-\lVert p(g)-p(b')\rVert\big| \le \lVert p(b)-p(b')\rVert$. Componiendo se obtiene la primera desigualdad. La segunda tiene dos partes:
- $D(g)\ge\kappa c$.
- $p$ actúa sobre cada par $(x_1,x_2)$ y $(y_1,y_2)$ con la matriz $\tfrac12\begin{psmallmatrix}1&1\\-1&1\end{psmallmatrix}=\tfrac1{\sqrt2}R_{45°}$, con $R_{45°}$ ortogonal. Por tanto $\lVert p(b)-p(b')\rVert=\lVert b-b'\rVert/\sqrt2$, y por lo mismo $p$ es biyectiva. ∎

**(c)** Si $b\neq g$, entonces $W>0$, y $W$ es diferenciable en $p(b)$ con $\nabla W = (p(b)-p(g))/W$, de norma 1. Por la regla de la cadena, $\nabla s = -(s/D)\,\nabla W$, cuya norma es $s/D>0$. ∎

**(d)** Por un lado $\ln s = -W/(\kappa(\sqrt A+c))$, y por otro $\ln s_0 = -W/(\kappa\sqrt A)$. Su cociente es $\sqrt A/(\sqrt A+c)=\rho$, así que $s=s_0^{\rho}$.

Al escalar por $\lambda$, $W(\lambda g,\lambda b)=\lambda W(g,b)$ y $\sqrt{A_{\lambda g}}=\lambda\sqrt{A_g}$, de modo que
$$s_c(\lambda g,\lambda b)=\exp\!\big(-\lambda W/(\kappa(\lambda\sqrt A+c))\big)=\exp\!\big(-W/(\kappa(\sqrt A+c/\lambda))\big)=s_{c/\lambda}(g,b).$$

Los límites se siguen de $\rho\to1$ (cuando $A\to\infty$) y de $\kappa(\sqrt A+c)\to\kappa c$ (cuando $A\to0$). ∎

**Proposición 2.** La intersección es $(a-t)a$ y la unión $2a^2-(a-t)a=a(a+t)$; derivar en $t=0$ da $-2/a$. Si $b\cap g=\varnothing$, una perturbación pequeña de $b$ la mantiene disjunta, así que el IoU es constante igual a 0 en un entorno y su gradiente es nulo. ∎

**Corolario 3.** Por (b), $s(g,b')\ge s(g,b)-\lVert b-b'\rVert/(\sqrt2\kappa c) > \tau+m-m=\tau$. ∎

### Verificación numérica

En `tests/test_original_model.py`, ejecutado en integración continua:

| Propiedad | Prueba |
|---|---|
| (a) rango e identidad | `test_rango_e_identidad` |
| $W$ euclídea en $p$ | `test_w2_es_euclidea_en_p` |
| (b) cociente máximo empírico frente a la cota, 4 combinaciones de $(\kappa,c)$ y 200 pares cada una | `test_cota_lipschitz_esquinas` |
| (c) gradiente no nulo donde el IoU es 0 | `test_gradiente_no_nulo_con_cajas_disjuntas` |
| (d) identidad de escala | `test_invarianza_escala_sin_piso_y_limite_nwd` |
| Calibración con el IoU | `test_calibracion_con_iou_en_objetos_grandes` |
| Proposición 2 en un tornillo de 8 px | `test_mas_tolerante_que_iou_en_tornillos_diminutos` |

Pendiente: la figura de sensibilidad (máximo cociente empírico de SGW y del IoU frente al tamaño del objeto, de 3 a 60 px), en `experiments/verificacion_teorema.py`.

### Honestidad teórica

El teorema habla de la **estabilidad de la asignación de etiquetas**, no del F1. Que una asignación más estable y con más positivos para los tornillos diminutos mejore el recall es la **hipótesis** que se pone a prueba, no algo demostrado.

Tres supuestos se violan o se debilitan en los datos reales:
- La caja es un modelo pobre de un tornillo visto en ángulo, que es elíptico.
- La asignación de baja calidad de RPN (paso 6 del pseudocódigo) da al menos un ancla a cada verdad tanto con IoU como con SGW, así que el efecto en RPN puede ser menor que en la cabeza RoI, que no tiene ese paso.
- Más positivos también significa positivos de peor calidad. Esto podría empeorar la precisión, y se mide.

## 4. Hipótesis del mecanismo y predicciones secundarias (D3, D4)

**H-M.** La asignación SGW corrige la ceguera ante tornillos pequeños (L) porque da a cada tornillo diminuto un número de positivos comparable al de los grandes y estable ante errores de 1–3 px (P, teorema 1 y proposición 2). Por tanto aumenta el recall del tercil pequeño y el F1 de instancia por grupo en test.

Predicciones declaradas antes de entrenar:
1. **Piloto, sin entrenar:** con IoU, el número de anclas y propuestas positivas por tornillo crece con el tamaño; con SGW la diferencia entre terciles se reduce. Si no se reduce, H-M es falsa desde el principio.
2. La ganancia en recall se concentra en el tercil pequeño. En el grande, la diferencia con Mask R-CNN debe ser menor que la del pequeño.
3. Ablación: quitar SGW de la cabeza RoI reduce la ganancia más que quitarlo de RPN, por la asignación de baja calidad de RPN (sección 3).
4. Robustez: al reducir la imagen al 75 % y al 50 % (tornillos más pequeños), la ventaja frente a Mask R-CNN crece.
5. Con $c\to0$ (sin piso), la ganancia en el tercil pequeño desaparece o se vuelve inestable entre semillas.

## 5. Criterio de éxito y de fracaso (D5) y hallazgo alternativo (E3 §14.5.9)

- **Éxito confirmatorio (R3):** superar a cada rival de la familia, con Holm, tres pruebas y diferencia mayor que δ.
- **Realidad:** δ = 0,20 se hereda del E1. Mask R-CNN tiene 0,726 de F1 por grupo, así que superarlo por δ exige llegar a 0,93, lo que no es plausible. Se declara desde ahora que el tope esperado es 4,0.
- **Hallazgo alternativo que sostiene el trabajo:** la prueba confirmatoria de la hipótesis del mecanismo, con su propia familia:
  - recall del tercil pequeño frente a Mask R-CNN, con un margen de 0,05 fijado aquí, antes de entrenar;
  - las predicciones 2–5;
  - el análisis de por qué la ceguera es compartida por la familia.

## 6. Piloto de asignación (2026-11-23, partición de desarrollo, sin entrenar)

Script: `experiments/piloto_asignacion.py`. Datos: los 3.530 tornillos de train de la partición agrupada; no se usaron val ni test. Para cada tornillo se mide:
- **RPN:** las anclas positivas, con umbral 0,7 y la asignación de baja calidad de torchvision;
- **RoI:** la fracción de 200 propuestas sintéticas positivas (umbral 0,5), generadas con un error gaussiano de σ = 2 px en cada esquina.

| Medida | anclas positivas RPN (peq. / med. / gr.) | propuestas positivas RoI (peq. / med. / gr.) |
|---|---|---|
| IoU | 1,35 / 1,22 / 1,13 | **0,44 / 0,77 / 0,98** |
| SGW κ = 0,225 | 1,36 / 1,41 / 1,13 | 0,23 / 0,44 / 0,85 |
| SGW κ = 0,45 | 1,50 / 1,45 / 1,18 | **0,85 / 0,98 / 1,00** |
| SGW κ = 0,9 | 6,09 / 10,36 / 7,26 | 1,00 / 1,00 / 1,00 |

**Lectura:**
- La predicción 1 **se cumple en la cabeza RoI y no en RPN**. Con IoU, un tornillo pequeño pierde más de la mitad de sus propuestas positivas por un error de 2 px. Con κ₀ la brecha entre terciles baja de 0,54 a 0,15.
- En RPN casi todos los tornillos, de cualquier tamaño, quedan con una sola ancla positiva con ambas medidas. Es la asignación de baja calidad, como anticipaba la sección 3, y refuerza la predicción 3.
- κ = 0,225 es más estricta que el IoU y κ = 0,9 satura: todo es positivo, también lo que es de mala calidad.
- La rejilla se mantiene como se declaró. No se cambia por este resultado: el piloto no mide F1.
- **Limitación del piloto:** σ = 2 px es un supuesto sobre el error de RPN, no una medida.
