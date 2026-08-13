import streamlit.components.v1 as components

def inject_hero_geometric_background():
    """
    Inyecta el fondo 'Hero Geometric' adaptado de React a Vanilla WebGL.
    Aplica un estilo claro (blanco/gris) con animaciones fluidas usando shader noise.
    """
    html_code = r"""
    <script>
    (function() {
        // Intentar acceder al documento padre (la app Streamlit principal)
        let parentDoc;
        try {
            parentDoc = window.parent.document;
        } catch (e) {
            console.error("No se pudo acceder al parent document.", e);
            return;
        }

        // Si ya existe, no duplicarlo
        if (parentDoc.getElementById('hero-geometric-container')) {
            return;
        }

        // Remover fondos anteriores si existen
        ['silk-aurora-container', 'pixel-canvas-container'].forEach(id => {
            const oldBg = parentDoc.getElementById(id);
            if (oldBg) oldBg.remove();
        });

        // Crear contenedor principal
        const container = parentDoc.createElement('div');
        container.id = 'hero-geometric-container';
        container.style.position = 'fixed';
        container.style.top = '0';
        container.style.left = '0';
        container.style.width = '100vw';
        container.style.height = '100vh';
        container.style.zIndex = '-1';
        container.style.pointerEvents = 'none'; // No bloquear interacciones
        container.style.backgroundColor = '#ffffff'; // Fondo base
        container.style.overflow = 'hidden';

        // Crear canvas
        const canvas = parentDoc.createElement('canvas');
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        canvas.style.display = 'block';
        container.appendChild(canvas);

        // Insertar al principio del app
        const appNode = parentDoc.querySelector('.stApp');
        if (appNode) {
            appNode.prepend(container);
        } else {
            parentDoc.body.prepend(container);
        }

        const gl = canvas.getContext("webgl", { antialias: false, alpha: true });
        if (!gl) {
            container.remove();
            return;
        }

        const VERTEX_SHADER = `
        attribute vec2 position;
        varying vec2 vUv;
        void main() {
            vUv = position * 0.5 + 0.5;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        `;

        const FRAGMENT_SHADER = `
        precision highp float;
        uniform float uTime;
        uniform vec2 uResolution;
        uniform vec3 uColor1;
        uniform vec3 uColor2;
        varying vec2 vUv;

        vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

        float snoise(vec2 v){
          const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                   -0.577350269189626, 0.024390243902439);
          vec2 i  = floor(v + dot(v, C.yy) );
          vec2 x0 = v -   i + dot(i, C.xx);
          vec2 i1;
          i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
          vec4 x12 = x0.xyxy + C.xxzz;
          x12.xy -= i1;
          i = mod(i, 289.0);
          vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
          + i.x + vec3(0.0, i1.x, 1.0 ));
          vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
          m = m*m ;
          m = m*m ;
          vec3 x = 2.0 * fract(p * C.www) - 1.0;
          vec3 h = abs(x) - 0.5;
          vec3 ox = floor(x + 0.5);
          vec3 a0 = x - ox;
          m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
          vec3 g;
          g.x  = a0.x  * x0.x  + h.x  * x0.y;
          g.yz = a0.yz * x12.xz + h.yz * x12.yw;
          return 130.0 * dot(m, g);
        }

        float random(vec2 st) {
            return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
        }

        void main() {
            vec2 uv = vUv;
            vec2 coord = gl_FragCoord.xy;
            
            float noise = snoise(uv * 1.5 + vec2(uTime * 0.05, uTime * 0.03)) * 0.25;
            float diagonal = (uv.x + uv.y) * 0.5;
            float gradient = diagonal * 1.2 + noise;
            
            vec3 deepBlue = uColor1;
            vec3 paleBlue = uColor2;
            vec3 softBlue = mix(deepBlue, paleBlue, 0.33);
            vec3 lightBlue = mix(deepBlue, paleBlue, 0.66);
            
            vec3 color;
            if (gradient < 0.3) {
                color = deepBlue;
            } else if (gradient < 0.55) {
                color = softBlue;
            } else if (gradient < 0.8) {
                color = lightBlue;
            } else {
                color = paleBlue;
            }
            
            // Dither simple basado en ruido para evitar banding en WebGL1
            float dither = random(coord) * 0.1;
            float threshold = fract(gradient * 4.0);
            
            if (gradient < 0.3 && threshold > dither * 5.0) {
                color = softBlue;
            } else if (gradient >= 0.3 && gradient < 0.55 && threshold > dither * 5.0) {
                color = lightBlue;
            } else if (gradient >= 0.55 && gradient < 0.8 && threshold > dither * 5.0) {
                color = paleBlue;
            }
            
            vec2 cornerDist = vec2(uv.x, uv.y);
            float fadeMask = smoothstep(0.0, 0.25, length(cornerDist));
            color = mix(vec3(1.0), color, fadeMask);
            
            float vignette = smoothstep(1.2, 0.3, length(uv - 0.5));
            color = mix(color, color * 0.95, (1.0 - vignette) * 0.3);
            
            // alpha 1.0 pero el canvas es transparente en el setup, asi que el color base es visible si falla
            gl_FragColor = vec4(color, 1.0);
        }
        `;

        function compileShader(type, source) {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                console.error("Shader compile error: ", gl.getShaderInfoLog(shader));
                return null;
            }
            return shader;
        }

        const vertexShader = compileShader(gl.VERTEX_SHADER, VERTEX_SHADER);
        const fragmentShader = compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER);

        if (!vertexShader || !fragmentShader) {
            container.remove(); // Si falla el shader, quitar el contenedor para revelar el fondo CSS blanco
            return;
        }

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            container.remove();
            return;
        }
        
        gl.useProgram(program);

        const position = gl.getAttribLocation(program, "position");
        const buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(position);
        gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

        const uTime = gl.getUniformLocation(program, "uTime");
        const uResolution = gl.getUniformLocation(program, "uResolution");
        const uColor1 = gl.getUniformLocation(program, "uColor1");
        const uColor2 = gl.getUniformLocation(program, "uColor2");

        function hexToRgb(hex) {
            hex = hex.replace("#", "");
            return [
                parseInt(hex.slice(0, 2), 16) / 255,
                parseInt(hex.slice(2, 4), 16) / 255,
                parseInt(hex.slice(4, 6), 16) / 255
            ];
        }

        // Colores base para Hero Geometric (tema claro/gris/blanco editorial)
        const color1 = hexToRgb("#e2e8f0"); // slate gray sutil
        const color2 = hexToRgb("#ffffff"); // blanco puro
        gl.uniform3f(uColor1, color1[0], color1[1], color1[2]);
        gl.uniform3f(uColor2, color2[0], color2[1], color2[2]);

        const resize = () => {
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            canvas.width = window.innerWidth * dpr;
            canvas.height = window.innerHeight * dpr;
            gl.viewport(0, 0, canvas.width, canvas.height);
            gl.uniform2f(uResolution, canvas.width, canvas.height);
        };
        window.addEventListener('resize', resize);
        resize();

        const start = performance.now();
        const speed = 1.0;

        function render(now) {
            const elapsed = (now - start) / 1000;
            gl.uniform1f(uTime, elapsed * speed);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
            requestAnimationFrame(render);
        }
        requestAnimationFrame(render);
    })();
    </script>
    """
    components.html(html_code, width=0, height=0)
