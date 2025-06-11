"use client"

import { useRef, useEffect } from "react"

export default function PlanetsGLSLVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const gl = canvas.getContext("webgl2")
    if (!gl) {
      console.error("WebGL2 not supported")
      return
    }

    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    const vertexShaderSource = `#version 300 es
      precision highp float;
      in vec4 a_position;
      void main() {
        gl_Position = a_position;
      }
    `

    const fragmentShaderSource = `#version 300 es
      precision highp float;
      out vec4 outColor;
      uniform vec2 u_resolution;
      uniform vec2 u_mouse; // Still available, though not explicitly used in this version
      uniform float u_time;

      // Helper: HSV to RGB conversion
      vec3 hsv(float h, float s, float v) {
        vec4 t = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
        vec3 p = abs(fract(vec3(h) + t.xyz) * 6.0 - vec3(t.w));
        return v * mix(vec3(t.x), clamp(p - vec3(t.x), 0.0, 1.0), s);
      }

      // Helper: Pseudo-random number generator
      float rand(vec2 n) { 
          return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
      }

      // Helper: Basic noise function
      float noise(vec2 p) {
          vec2 ip = floor(p);
          vec2 u = fract(p);
          u = u*u*(3.0-2.0*u); // Smoothstep
          
          float res = mix(
              mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
              mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),
              u.y);
          return res*res; 
      }

      // Helper: Fractional Brownian Motion for more complex patterns
      float fbm(vec2 p) {
          float v = 0.0;
          float a = 0.5;
          vec2 shift = vec2(100.0);
          // Use a rotation matrix to break up axial alignment
          mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5)); 
          for (int i = 0; i < 5; ++i) { 
              v += a * noise(p);
              p = rot * p * 2.0 + shift;
              a *= 0.5;
          }
          return v;
      }

      void main() {
        vec2 r_screen = u_resolution;
        vec2 FC_screen = gl_FragCoord.xy;
        float t = u_time;
        vec4 final_color;

        // --- 1. Galaxy Background ---
        vec2 uv_screen = FC_screen / r_screen;
        
        // Stars
        float stars = 0.0;
        float star_density = 0.996; 
        // Check for star, make it twinkle
        if (rand(uv_screen * 250.0 + floor(t*2.0)) > star_density) { // Scale uv for smaller stars, animate seed for twinkle
            stars = pow(rand(uv_screen * 500.0 + floor(t*5.0)), 10.0) * 0.8 + 0.2; 
        }
        
        // Nebula clouds
        float nebula_pattern = fbm(uv_screen * 2.5 + vec2(sin(t*0.05), cos(t*0.05)) + 0.5); // Slow moving nebula
        vec3 nebula_color1 = vec3(0.05, 0.05, 0.2); // Deep blue
        vec3 nebula_color2 = vec3(0.3, 0.1, 0.4);   // Purple/magenta
        vec3 nebula_color = mix(nebula_color1, nebula_color2, smoothstep(0.3, 0.7, nebula_pattern));
        
        final_color.rgb = nebula_color + vec3(stars); // Add stars to nebula
        final_color.a = 1.0;

        // --- 2. Scope Windows ---
        const int NUM_WINDOWS = 3;
        vec2 window_centers_norm[NUM_WINDOWS]; // Normalized screen coordinates (0-1)
        window_centers_norm[0] = vec2(0.25, 0.65);
        window_centers_norm[1] = vec2(0.75, 0.35);
        window_centers_norm[2] = vec2(0.50, 0.70); // Slightly overlapping

        float window_radius_norm = 0.16; // Radius relative to min screen dimension
        float window_radius_px = window_radius_norm * min(r_screen.x, r_screen.y);

        for (int k = 0; k < NUM_WINDOWS; ++k) {
            vec2 window_center_px = window_centers_norm[k] * r_screen;
            float dist_from_window_center = length(FC_screen - window_center_px);

            if (dist_from_window_center < window_radius_px) {
                // Pixel is inside this scope window
                vec4 o_scope = vec4(0,0,0,1);
                
                // Transform FC_screen to be local to this window's "virtual screen"
                // fc_in_scope: coordinates from (0,0) at bottom-left of scope, to (2*radius, 2*radius) at top-right
                vec2 fc_in_scope = FC_screen - (window_center_px - window_radius_px); 
                vec2 r_scope = vec2(2.0 * window_radius_px); // "Resolution" of this scope window

                // Original visualization logic, adapted for this scope
                float g_scope=0., e_scope=0., s_scope=0.; // Scope-local variables
                for(float i=0.; ++i<70.;){ // Reduced iterations slightly for performance with multiple scopes
                  // The 'p' vector calculation, using fc_in_scope and r_scope
                  vec3 p_scope = vec3(((fc_in_scope * 2.0 - r_scope) / r_scope.x) * 0.7 + vec2(0,1), g_scope + 0.1);
                  
                  // Vary rotation and hue per scope for visual distinction
                  float time_offset = float(k) * 2.1; // Make scopes animate differently
                  float hue_offset = float(k) * 0.27;
                  p_scope.zx *= mat2(cos(t*.3 + time_offset), sin(t*.3 + time_offset), 
                                    -sin(t*.3 + time_offset), cos(t*.3 + time_offset));
                  s_scope=2.;
                  for(int j=0;j++<15;p_scope=vec3(2,5,2)-abs(abs(p_scope)*e_scope-vec3(4))) // Reduced inner loop slightly
                    s_scope*=e_scope=max(1.01,13./dot(p_scope,p_scope));
                  g_scope+=mod(length(p_scope.xz),p_scope.y*5.)/s_scope;
                  s_scope=log(s_scope*.1);
                  o_scope.rgb+=hsv(g_scope + hue_offset, p_scope.z, s_scope/6e2);
                }

                // Vignetting for this scope window
                float vignette_dist_norm = dist_from_window_center / window_radius_px; // 0 at center, 1 at edge
                float vignette_start = 0.5; 
                float vignette_strength = 0.95;
                float vignette_effect = smoothstep(vignette_start, 1.0, vignette_dist_norm);
                o_scope.rgb *= (1.0 - vignette_effect * vignette_strength);
                
                final_color = o_scope; // This scope takes precedence
                break; 
            }
        }
        outColor = final_color;
      }
    `

    const createShader = (gl: WebGL2RenderingContext, type: number, source: string) => {
      const shader = gl.createShader(type)
      if (!shader) {
        console.error("Failed to create shader")
        return null
      }
      gl.shaderSource(shader, source)
      gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("Shader compilation error:", gl.getShaderInfoLog(shader))
        gl.deleteShader(shader)
        return null
      }
      return shader
    }

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource)
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource)

    if (!vertexShader || !fragmentShader) return

    const program = gl.createProgram()
    if (!program) {
      console.error("Failed to create program")
      return
    }
    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program linking error:", gl.getProgramInfoLog(program))
      return
    }

    const positionBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW)
    const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
    gl.enableVertexAttribArray(positionAttributeLocation)
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0)

    const resolutionUniformLocation = gl.getUniformLocation(program, "u_resolution")
    const timeUniformLocation = gl.getUniformLocation(program, "u_time")
    // const mouseUniformLocation = gl.getUniformLocation(program, "u_mouse") // Mouse not used in this shader version

    const startTime = Date.now()

    const render = () => {
      const currentTime = (Date.now() - startTime) / 1000
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.clearColor(0, 0, 0, 1) // Clear to black, galaxy will overwrite
      gl.clear(gl.COLOR_BUFFER_BIT)

      if (resolutionUniformLocation) gl.uniform2f(resolutionUniformLocation, canvas.width, canvas.height)
      if (timeUniformLocation) gl.uniform1f(timeUniformLocation, currentTime)
      // if (mouseUniformLocation) gl.uniform2f(mouseUniformLocation, mouseX, mouseY); // If mouse interaction is re-added

      gl.useProgram(program)
      gl.drawArrays(gl.TRIANGLES, 0, 6)
      requestAnimationFrame(render)
    }
    render()

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      if (program) gl.deleteProgram(program)
      if (vertexShader) gl.deleteShader(vertexShader)
      if (fragmentShader) gl.deleteShader(fragmentShader)
    }
  }, [])

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh", backgroundColor: "black" }}>
      <canvas
        ref={canvasRef}
        style={{ display: "block", width: "100vw", height: "100vh", position: "fixed", top: 0, left: 0 }}
      />
      <div
        style={{
          position: "fixed",
          bottom: "20px",
          right: "20px",
          color: "white",
          fontSize: "14px",
          fontFamily: "monospace",
          textShadow: "1px 1px 2px rgba(0,0,0,0.8)",
          zIndex: 1000,
        }}
      >
        <a
          href="https://x.com/YoheiNishitsuji/status/1928436240199619008"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "white", textDecoration: "none" }}
        >
          Original GLSL by @Yohei Nishitsuji
        </a>
      </div>
    </div>
  )
}
