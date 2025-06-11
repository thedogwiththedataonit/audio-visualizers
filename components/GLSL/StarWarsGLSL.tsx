"use client"

import { useRef, useEffect, useState, useCallback } from "react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { X, Settings } from "lucide-react"

interface AudioData {
  level: number
  bassLevel: number
  midLevel: number
  trebleLevel: number
  frequencyData: Float32Array
  waveformData: Float32Array
}

export default function StarWarsGLSLVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  // UI state
  const [showControls, setShowControls] = useState(false)
  const [showFps, setShowFps] = useState(false)
  const fpsRef = useRef<HTMLDivElement>(null)
  const lastTimeRef = useRef<number>(0)
  const frameCountRef = useRef<number>(0)

  // Audio state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.7)
  const [trackName, setTrackName] = useState("")
  const [isDragOver, setIsDragOver] = useState(false)

  // Audio control state
  const [audioIntensity, setAudioIntensity] = useState(0.5)
  const [bassInfluence, setBassInfluence] = useState(0.5)
  const [midInfluence, setMidInfluence] = useState(0.5)
  const [trebleInfluence, setTrebleInfluence] = useState(0.5)
  const [tunnelSpeed, setTunnelSpeed] = useState(1.0)
  const [spiralIntensity, setSpiralIntensity] = useState(1.0)
  const [colorMode, setColorMode] = useState("default")
  const [smoothing, setSmoothing] = useState(0.85)

  // Audio refs
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null)

  // Audio data arrays
  const frequencyDataRef = useRef(new Float32Array(64))
  const waveformDataRef = useRef(new Float32Array(32))
  const smoothedAudioDataRef = useRef<AudioData>({
    level: 0,
    bassLevel: 0,
    midLevel: 0,
    trebleLevel: 0,
    frequencyData: new Float32Array(64),
    waveformData: new Float32Array(32)
  })

  // WebGL refs
  const programRef = useRef<WebGLProgram | null>(null)
  const resolutionUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const timeUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  
  // Audio uniform locations
  const audioLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const bassLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const midLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const trebleLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const frequencyDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const waveformDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const audioIntensityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const bassInfluenceUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const midInfluenceUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const trebleInfluenceUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const tunnelSpeedUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const spiralIntensityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const colorModeUniformLocationRef = useRef<WebGLUniformLocation | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Get WebGL2 context
    const gl = canvas.getContext("webgl2")
    if (!gl) {
      console.error("WebGL2 not supported")
      return
    }

    // Resize canvas to match window size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Vertex shader source
    const vertexShaderSource = `#version 300 es
      precision highp float;
      in vec4 a_position;
      void main() {
        gl_Position = a_position;
      }
    `

    // Fragment shader source - preserving the exact GLSL code structure with explanations
    const fragmentShaderSource = `#version 300 es
  precision highp float;
  out vec4 outColor;
  uniform vec2 u_resolution;  // Screen resolution (width, height)
  uniform float u_time;       // Time in seconds

  // HSV to RGB color conversion function
  // h: hue (0-1), s: saturation (0-1), v: value/brightness (0-1)
  // This is a compact implementation of the HSV color space conversion
  vec3 hsv(float h,float s,float v){
    vec4 t=vec4(1.,2./3.,1./3.,3.);  // Constants for HSV conversion
    vec3 p=abs(fract(vec3(h)+t.xyz)*6.-vec3(t.w));  // Color channel calculations
    return v*mix(vec3(t.x),clamp(p-vec3(t.x),0.,1.),s);  // Final RGB blend
  }

  void main() {
    vec2 r = u_resolution;                // Screen resolution
    vec2 FC = gl_FragCoord.xy;            // Current pixel coordinates
    float t = u_time;                     // Current time
    vec4 o = vec4(0,0,0,1);               // Output color (starts black, fully opaque)
    
    // Variable declarations:
    // i: loop counter for ray steps
    // e: distance estimation value
    // R: radius/length of position vector
    // s: scale factor for fractal iteration
    float i,e,R,s;
    
    // q: current ray position
    // p: transformed position for fractal evaluation
    // d: ray direction vector (normalized screen coordinates with depth component)
    vec3 q,p,d=vec3((FC.xy-r*.5)/r,.1);   // Ray direction from screen center with z=0.1
    
    // Ray marching loop - iterates 99 times to create the visual effect
    // This is essentially a distance field raymarcher
    for(q.zx--;i++<99.;){
      // Accumulate color based on distance estimation (e) and scale (s)
      // This creates a glow effect where the ray gets close to the surface
      // The color is in the yellow/orange spectrum (hue 0.1)
      o.rgb+=hsv(.1,e,min(e*s,.4-e)/25.);
      
      // Reset scale factor for the inner loop
      s=2.;
      
      // Move ray forward by distance e*R*0.7 (safe step distance)
      // This is the core of the raymarching algorithm
      p=q+=d*e*R*.7;
      
      // Transform position into a logarithmic spiral space
      // This creates the spiral tunnel effect with time-based rotation
      // - log2(R) creates logarithmic distance scaling
      // - exp2(0.9-p.z/R) creates exponential radial scaling
      // - atan(p.y,p.x) gives the angular component with cosine modulation
      p=vec3(log2(R=length(p))-t*.5,exp2(.9-p.z/R),atan(p.y,p.x)+cos(t*.5)*.1);
      
      // Fractal noise generation through frequency doubling (octave summation)
      // This is similar to a simplified Perlin noise implementation
      // The loop doubles the frequency (s+=s) each iteration, creating multi-scale detail
      // The dot product between cosine waves creates interference patterns
      for(e=--p.y;s<6e2;s+=s)
        // This line generates fractal noise through:
        // 1. Cosine waves in different dimensions (p.yzy*s)
        // 2. Interference patterns via dot product
        // 3. Frequency scaling (1/s) for proper octave weighting
        // The result is a complex noise pattern with both high and low frequency details
        e+=dot(cos(p.yzy*s)-.5,1.-cos(p*s))/s*.3;
    }
    
    // Final color output
    outColor = o;
  }
`

    // Create and compile shaders
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

    // Create program and link shaders
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

    // Use our shader program - Moved outside render
    gl.useProgram(program)

    // Set up position buffer
    const positionBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
      gl.STATIC_DRAW,
    )

    // Set up position attribute
    const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
    gl.enableVertexAttribArray(positionAttributeLocation)
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0)

    // Get uniform locations
    const resolutionUniformLocation = gl.getUniformLocation(program, "u_resolution")
    const timeUniformLocation = gl.getUniformLocation(program, "u_time")

    // Start time for animation
    const startTime = performance.now()

    // Render function
    const render = () => {
      // Update time uniform
      const currentTime = (performance.now() - startTime) / 1000

      // Set viewport and clear
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)

      // Update uniforms
      gl.uniform2f(resolutionUniformLocation, canvas.width, canvas.height)
      gl.uniform1f(timeUniformLocation, currentTime)

      // Draw
      gl.drawArrays(gl.TRIANGLES, 0, 6)

      // Request next frame
      requestAnimationFrame(render)
    }

    // Start rendering
    render()

    // Cleanup
    return () => {
      window.removeEventListener("resize", resizeCanvas)
      gl.deleteProgram(program)
      gl.deleteShader(vertexShader)
      gl.deleteShader(fragmentShader)
      gl.deleteBuffer(positionBuffer)
      gl.useProgram(null)
    }
  }, [])

  return (
    <div className="relative w-full h-screen">
      <canvas
        ref={canvasRef}
        style={{
          display: "block",
          width: "100%",
          height: "100vh",
          position: "absolute",
          top: 0,
          left: 0,
        }}
      />
      <a
        href="https://x.com/YoheiNishitsuji/status/1915756430084256054"
        target="_blank"
        rel="noopener noreferrer"
        className="absolute bottom-4 right-4 text-white text-sm opacity-70 hover:opacity-100 transition-opacity z-10"
      >
        @Yohei Nishitsuji
      </a>
    </div>
  )
}
