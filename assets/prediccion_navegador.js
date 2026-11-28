/*
 * Predicción en vivo en el navegador (modo navegador, WDSD_INFERENCIA_VIVO=0).
 *
 * Por qué: una predicción con ONNX Runtime en el servidor sube el proceso a ~0,9 GB y el plan gratuito de Render da
 * 512 MB. Aquí el servidor solo entrega el mismo artefacto (/modelo/<nombre>.onnx, 84 MB, se descarga una vez y queda
 * en la caché del navegador) y onnxruntime-web 1.20.1 (assets/vendor/ort/, WebAssembly) lo ejecuta en el equipo de
 * quien abre el dashboard. Mismo grafo y mismos pesos que src/inference.py.
 *
 * Reparto del trabajo: Python (pages/modelos.py:nav_imagen y vivo) decodifica, valida, da el aviso fuera de dominio y
 * redimensiona a 1280x720 con Pillow, exactamente como en modo servidor, y pasa esa imagen en PNG (sin pérdida) a
 * esta función. Así el modelo recibe los mismos píxeles en los dos modos y da las mismas detecciones.
 */
(function () {
  "use strict";

  var H = 720, W = 1280, NARANJA = [230, 126, 34], ALFA = 140 / 255;
  var estado = {cargandoOrt: null, sesion: null, modelo: null, ocupado: false, hilos: 1};

  // --- componentes de Dash como JSON (los pinta React, igual que si vinieran de Python) ---
  function c(tipo, props, hijos) {
    var p = Object.assign({}, props || {});
    if (hijos !== undefined) p.children = hijos;
    return {type: tipo, namespace: "dash_html_components", props: p};
  }
  function aviso(color, texto) { return c("Div", {className: "alert alert-" + color + " mt-2"}, texto); }
  function fila(th, td) { return c("Tr", {}, [c("Th", {}, th), c("Td", {}, td)]); }
  function mostrar(hijos) { window.dash_clientside.set_props("nav-resultado", {children: hijos}); }
  function coma(x, d) { return x.toFixed(d).replace(".", ","); }

  function cargarOrt(cfg) {
    if (window.ort) return Promise.resolve(window.ort);
    if (!estado.cargandoOrt) {
      estado.cargandoOrt = new Promise(function (ok, mal) {
        var s = document.createElement("script");
        s.src = cfg.ort_js;
        s.onload = function () { ok(window.ort); };
        s.onerror = function () { estado.cargandoOrt = null; mal(new Error("no se pudo cargar onnxruntime-web")); };
        document.head.appendChild(s);
      });
    }
    return estado.cargandoOrt.then(function (ort) {
      ort.env.wasm.wasmPaths = new URL(cfg.ort_dir, window.location.href).href;
      // Los hilos de WebAssembly exigen aislamiento de origen cruzado (cabeceras COOP/COEP de app.py).
      estado.hilos = window.crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;
      ort.env.wasm.numThreads = estado.hilos;
      ort.env.logLevel = "error";  // como log_severity_level=3 en Python: la forma fija de salida es solo nominal
      return ort;
    });
  }

  async function descargar(url, mb, nombre) {
    var r = await fetch(url);
    if (!r.ok) throw new Error("el servidor respondió " + r.status + " al pedir el modelo");
    var total = +r.headers.get("Content-Length") || mb * 1048576;
    var lector = r.body.getReader(), partes = [], recibido = 0, ultimo = -1;
    for (;;) {
      var paso = await lector.read();
      if (paso.done) break;
      partes.push(paso.value);
      recibido += paso.value.length;
      var pct = Math.floor(100 * recibido / total);
      if (pct >= ultimo + 5) {
        ultimo = pct;
        mostrar(aviso("info", "Descargando " + nombre + ": " + Math.round(recibido / 1048576) + " de " +
          Math.round(total / 1048576) + " MB (solo la primera vez)…"));
      }
    }
    var buf = new Uint8Array(recibido), o = 0;
    partes.forEach(function (p) { buf.set(p, o); o += p.length; });
    return buf;
  }

  async function sesion(ort, cfg, modelo, nombre) {
    if (estado.modelo === modelo) return estado.sesion;
    if (estado.sesion) {  // un solo modelo en memoria, como en el servidor
      await estado.sesion.release();
      estado.sesion = null; estado.modelo = null;
    }
    var m = cfg.modelos[modelo];
    var bytes = await descargar(m.url, m.mb, nombre);
    mostrar(aviso("info", "Preparando " + nombre + " en tu navegador…"));
    estado.sesion = await ort.InferenceSession.create(bytes, {executionProviders: ["wasm"], logSeverityLevel: 3});
    estado.modelo = modelo;
    return estado.sesion;
  }

  function lienzo(w, h) {
    var cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    return cv;
  }

  // Superposición: igual que src/explain.py:superponer (máscara > 0,5 en naranja, alfa 140, caja y score).
  function superponer(px, cajas, scores, mascaras, keep) {
    var cv = lienzo(W, H), ctx = cv.getContext("2d"), img = ctx.createImageData(W, H), N = W * H;
    img.data.set(px);
    var union = new Uint8Array(N);
    keep.forEach(function (k) {
      var off = k * N;
      for (var i = 0; i < N; i++) if (mascaras[off + i] > 0.5) union[i] = 1;
    });
    for (var i = 0; i < N; i++) if (union[i]) for (var ch = 0; ch < 3; ch++) {
      var j = 4 * i + ch;
      img.data[j] = Math.round(img.data[j] * (1 - ALFA) + NARANJA[ch] * ALFA);
    }
    ctx.putImageData(img, 0, 0);
    ctx.lineWidth = 2; ctx.strokeStyle = "rgb(230,126,34)"; ctx.fillStyle = "#fff"; ctx.font = "12px sans-serif";
    keep.forEach(function (k) {
      var x1 = cajas[4 * k], y1 = cajas[4 * k + 1], x2 = cajas[4 * k + 2], y2 = cajas[4 * k + 3];
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillText(scores[k].toFixed(2), x1, Math.max(10, y1 - 3));
    });
    return cv.toDataURL("image/jpeg", 0.85);
  }

  // Explicación local: igual que src/explain.py:mapa_confianza (máximo de probabilidad de máscara × score).
  function mapaConfianza(scores, mascaras, keep) {
    var cv = lienzo(W, H), ctx = cv.getContext("2d"), img = ctx.createImageData(W, H), N = W * H;
    var m = new Float32Array(N);
    keep.forEach(function (k) {
      var off = k * N, s = scores[k];
      for (var i = 0; i < N; i++) { var v = mascaras[off + i] * s; if (v > m[i]) m[i] = v; }
    });
    for (var i = 0; i < N; i++) {
      img.data[4 * i] = m[i] * NARANJA[0]; img.data[4 * i + 1] = m[i] * NARANJA[1];
      img.data[4 * i + 2] = m[i] * NARANJA[2]; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return cv.toDataURL("image/jpeg", 0.85);
  }

  async function predecir(imagen, modelo, cfg) {
    if (!cfg || cfg.modo !== "navegador" || !modelo || !cfg.modelos[modelo]) {
      return window.dash_clientside.no_update;
    }
    if (!imagen) return null;  // entrada inválida: el motivo lo muestra Python
    if (estado.ocupado) return aviso("warning", "Espera a que termine la predicción anterior y vuelve a intentarlo.");
    var bmp = await createImageBitmap(await (await fetch(imagen)).blob(),
                                      {colorSpaceConversion: "none", premultiplyAlpha: "none"});

    estado.ocupado = true;
    var nombre = cfg.modelos[modelo].nombre;
    try {
      mostrar(aviso("info", "Cargando el motor de inferencia (onnxruntime-web)…"));
      var ort = await cargarOrt(cfg);
      var s = await sesion(ort, cfg, modelo, nombre);

      // Preprocesado de src/inference.py:predecir: la imagen ya viene a 1280x720 (Pillow); CHW float32 en [0, 1].
      var cv = lienzo(W, H), ctx = cv.getContext("2d");
      ctx.drawImage(bmp, 0, 0);
      bmp.close();
      var px = ctx.getImageData(0, 0, W, H).data, N = W * H, x = new Float32Array(3 * N);
      for (var i = 0; i < N; i++) {
        x[i] = px[4 * i] / 255; x[N + i] = px[4 * i + 1] / 255; x[2 * N + i] = px[4 * i + 2] / 255;
      }
      mostrar(aviso("info", "Prediciendo con " + nombre + " en tu navegador (" + estado.hilos +
        (estado.hilos === 1 ? " hilo" : " hilos") + "; tarda entre 5 y 20 s)…"));
      var t0 = performance.now();
      var o = await s.run({imagen: new ort.Tensor("float32", x, [3, H, W])});
      var ms = performance.now() - t0;

      var scores = o.scores.data, u = cfg.modelos[modelo].umbral, keep = [];
      for (var k = 0; k < scores.length; k++) if (scores[k] >= u) keep.push(k);
      var n = keep.length, lo = Math.max(0, n - cfg.radio), hi = n + cfg.radio;
      var orden = keep.map(function (k) { return scores[k]; }).sort(function (a, b) { return b - a; }).slice(0, 15);
      var res = [
        c("Div", {className: "row"}, [
          c("Div", {className: "col-lg-6"}, [c("H6", {}, "Máscaras predichas"),
            c("Img", {src: superponer(px, o.boxes.data, scores, o.masks.data, keep), className: "img-fluid",
                      alt: "Imagen con las máscaras de tornillo predichas en naranja"})]),
          c("Div", {className: "col-lg-6"}, [c("H6", {}, "Explicación local: mapa de confianza por píxel"),
            c("Img", {src: mapaConfianza(scores, o.masks.data, keep), className: "img-fluid",
                      alt: "Mapa de confianza: probabilidad de máscara por score de cada detección"})]),
        ]),
        c("Table", {className: "table table-sm table-bordered mt-2"}, c("Tbody", {}, [
          fila("Tornillos detectados", String(n)),
          fila("Intervalo conforme 90 % del número de tornillos",
               lo + " a " + hi + " (cobertura empírica en test: " + Math.round(100 * cfg.cobertura) + " %)"),
          fila("Umbral de score (elegido en val)", coma(u, 2)),
          fila("Latencia (tu navegador, onnxruntime-web, " + estado.hilos + (estado.hilos === 1 ? " hilo)" : " hilos)"),
               Math.round(ms) + " ms"),
          fila("Scores de las detecciones", orden.map(function (v) { return coma(v, 2); }).join("; ") || "—"),
        ])),
        c("Small", {className: "text-muted"}, "Calculado en tu navegador con el mismo archivo ONNX que usa el " +
          "servidor en local. El score de detección no está calibrado como probabilidad (ECE del E2); el intervalo " +
          "del conteo sí tiene garantía de cobertura bajo intercambiabilidad, que entre escenas no se cumple del todo."),
      ];
      Object.keys(o).forEach(function (k) { if (o[k].dispose) o[k].dispose(); });
      return res;
    } catch (e) {
      console.error("[WDSD] predicción en el navegador:", e);
      var msg = String((e && e.message) || e);
      var memoria = /memory|RangeError|allocation|OOM/i.test(msg);
      return aviso("danger", memoria
        ? "Tu navegador se quedó sin memoria (la predicción necesita ~1,3 GB). Prueba en un computador con Chrome, " +
          "Edge o Firefox, o usa la galería precalculada."
        : "No se pudo ejecutar el modelo en tu navegador (" + msg + "). Usa la galería precalculada.");
    } finally {
      estado.ocupado = false;
    }
  }

  window.dash_clientside = window.dash_clientside || {};
  window.dash_clientside.wdsd = {predecir: predecir};
})();
