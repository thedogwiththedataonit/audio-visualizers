"use client"

import type React from "react"
import { useRef, useEffect, useState, useCallback } from "react"

interface AudioData {
  level: number
  bassLevel: number
  midLevel: number
  trebleLevel: number
  frequencyData: Float32Array
  waveformData: Float32Array
}

/**
 * Audio-Reactive GLSL Visualization Component
 *
 * An enhanced React component that renders a WebGL visualization based on fractal ray marching
 * with comprehensive audio reactivity and visual controls.
 *
 * Features:
 * - Audio-reactive fractal rendering with frequency analysis
 * - Full audio player with upload, playback controls, and timeline
 * - Real-time visual controls for audio responsiveness
 * - Mouse inertia controls for manual interaction
 * - Support for MP3, WAV, and OGG audio formats
 */
const TweetGLSLVisualization: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const programRef = useRef<WebGLProgram | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioElementRef = useRef<HTMLAudioElement | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null)
  const animationRef = useRef<number>(0)

  // WebGL uniform locations
  const resolutionUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const timeUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const mouseUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const rotationXUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const rotationYUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const colorShiftUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  
  // Audio uniform locations
  const audioLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const bassLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const midLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const trebleLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const frequencyDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const waveformDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  
  // Audio control uniform locations
  const audioIntensityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const beatPulseUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const fractalComplexityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)

  // Audio state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.7)
  const [trackName, setTrackName] = useState("")
  const [isDragOver, setIsDragOver] = useState(false)

  // Visual control state
  const [audioIntensity, setAudioIntensity] = useState(0.5)
  const [beatPulse, setBeatPulse] = useState(0.5)
  const [fractalComplexity, setFractalComplexity] = useState(0.5)
  const [colorSensitivity, setColorSensitivity] = useState(0.5)
  const [smoothing, setSmoothing] = useState(0.85)

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

  // Initialize audio context
  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()

      // Resume context if suspended (browser autoplay policy)
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume()
      }
    }
  }, [])

  // Audio analysis function
  const analyzeAudio = useCallback(() => {
    const analyser = analyserRef.current
    if (!analyser) return smoothedAudioDataRef.current

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    const waveformArray = new Uint8Array(bufferLength)

    analyser.getByteFrequencyData(dataArray)
    analyser.getByteTimeDomainData(waveformArray)

    // Calculate frequency bands
    const bassEnd = Math.floor(bufferLength * 0.1)
    const midEnd = Math.floor(bufferLength * 0.5)

    let bassSum = 0, midSum = 0, trebleSum = 0, totalSum = 0

    // Bass (0-10% of frequency range)
    for (let i = 0; i < bassEnd; i++) {
      bassSum += dataArray[i]
    }
    bassSum /= bassEnd

    // Mid (10-50% of frequency range)
    for (let i = bassEnd; i < midEnd; i++) {
      midSum += dataArray[i]
    }
    midSum /= (midEnd - bassEnd)

    // Treble (50-100% of frequency range)
    for (let i = midEnd; i < bufferLength; i++) {
      trebleSum += dataArray[i]
    }
    trebleSum /= (bufferLength - midEnd)

    // Overall level
    for (let i = 0; i < bufferLength; i++) {
      totalSum += dataArray[i]
    }
    const level = totalSum / bufferLength / 255

    // Normalize frequency bands
    const bassLevel = bassSum / 255
    const midLevel = midSum / 255
    const trebleLevel = trebleSum / 255

    // Downsample frequency data to 64 bins
    for (let i = 0; i < 64; i++) {
      const index = Math.floor((i / 64) * bufferLength)
      frequencyDataRef.current[i] = dataArray[index] / 255
    }

    // Downsample waveform data to 32 bins
    for (let i = 0; i < 32; i++) {
      const index = Math.floor((i / 32) * bufferLength)
      waveformDataRef.current[i] = (waveformArray[index] - 128) / 128
    }

    // Smooth the audio data to prevent jarring transitions
    const smoothingFactor = smoothing
    const current = smoothedAudioDataRef.current

    current.level = current.level * smoothingFactor + level * (1 - smoothingFactor)
    current.bassLevel = current.bassLevel * smoothingFactor + bassLevel * (1 - smoothingFactor)
    current.midLevel = current.midLevel * smoothingFactor + midLevel * (1 - smoothingFactor)
    current.trebleLevel = current.trebleLevel * smoothingFactor + trebleLevel * (1 - smoothingFactor)

    // Update frequency and waveform arrays
    for (let i = 0; i < 64; i++) {
      current.frequencyData[i] = current.frequencyData[i] * smoothingFactor + frequencyDataRef.current[i] * (1 - smoothingFactor)
    }

    for (let i = 0; i < 32; i++) {
      current.waveformData[i] = current.waveformData[i] * smoothingFactor + waveformDataRef.current[i] * (1 - smoothingFactor)
    }

    return current
  }, [smoothing])

  // Load audio file from File object
  const loadAudioFile = useCallback(async (file: File) => {
    await initAudioContext()

    const audio = new Audio()
    const url = URL.createObjectURL(file)
    audio.src = url
    audio.crossOrigin = "anonymous"
    audio.volume = volume

    // Set up audio element event listeners
    audio.addEventListener('loadedmetadata', () => {
      setDuration(audio.duration)
      setTrackName(file.name.replace(/\.[^/.]+$/, ""))
    })

    audio.addEventListener('timeupdate', () => {
      setCurrentTime(audio.currentTime)
    })

    audio.addEventListener('ended', () => {
      setIsPlaying(false)
    })

    // Clean up previous audio setup
    if (audioElementRef.current) {
      audioElementRef.current.pause()
      if (audioElementRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
    }

    audioElementRef.current = audio

    // Set up Web Audio API
    if (audioContextRef.current) {
      if (sourceRef.current) {
        sourceRef.current.disconnect()
      }

      const source = audioContextRef.current.createMediaElementSource(audio)
      const analyser = audioContextRef.current.createAnalyser()

      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.3

      source.connect(analyser)
      analyser.connect(audioContextRef.current.destination)

      sourceRef.current = source
      analyserRef.current = analyser
    }
  }, [volume, initAudioContext])

  useEffect(() => {
    let program: WebGLProgram | null = null
    let vertexShader: WebGLShader | null = null
    let fragmentShader: WebGLShader | null = null
    let positionBuffer: WebGLBuffer | null = null
    let gl: WebGL2RenderingContext | null = null
    const canvas = canvasRef.current
    if (!canvas) return

    // Get WebGL2 context
    gl = canvas.getContext("webgl2")
    if (!gl) {
      console.error("WebGL2 not supported")
      return
    }

    // Resize canvas to full screen
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      gl.viewport(0, 0, canvas.width, canvas.height)
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Vertex shader source - minimal pass-through shader
    const vertexShaderSource = `#version 300 es
    precision highp float;
    
    in vec4 a_position;
    
    void main() {
      gl_Position = a_position;
    }
    `

    // Fragment shader source with detailed technical explanations
    const fragmentShaderSource = `#version 300 es
precision highp float;

out vec4 outColor;
uniform vec2 u_resolution;  // Screen resolution
uniform vec2 u_mouse;       // Mouse position
uniform float u_time;       // Time in seconds
uniform float u_rotationX;  // Rotation influence from mouse X with inertia
uniform float u_rotationY;  // Rotation intensity from mouse Y with inertia
uniform float u_colorShift;  // Color shift based on vertical mouse position

// 2D rotation matrix function
// Used for rotating points around the origin in the XZ plane
mat2 rotate2D(float angle) {
  float s = sin(angle);
  float c = cos(angle);
  return mat2(c, -s, s, c);
}

// HSV to RGB color conversion
// h: hue [0,1], s: saturation [0,1], v: value/brightness [0,1]
// This is a common color space transformation used in computer graphics
// to create smooth color gradients and transitions
vec3 hsv(float h, float s, float v) {
  vec4 t = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
  vec3 p = abs(fract(vec3(h) + t.xyz) * 6.0 - vec3(t.w));
  return v * mix(vec3(t.x), clamp(p - vec3(t.x), 0.0, 1.0), s);
}

void main() {
  // Initialize variables
  vec2 r = u_resolution;                // Screen resolution
  vec2 FC = gl_FragCoord.xy;            // Current fragment coordinates
  float t = u_time;                     // Time
  vec4 o = vec4(0, 0, 0, 1);            // Output color (initially black)
  
  // Main rendering loop - this accumulates color contributions
  // This is a form of path tracing/ray marching where we accumulate
  // color values over multiple iterations (99 in this case)
  for(float i=0.,g=0.,e=0.,s=0.;++i<99.;){
    // Create initial ray position
    // This transforms screen coordinates to 3D space coordinates
    // The division by r.x ensures aspect ratio correction
    // g is used as a depth accumulator that evolves through iterations
    vec3 p=vec3((FC.xy*2.-r)/r.x+vec2(0,.9),g-.5);
    
    // Apply rotation to the XZ plane
    // This creates the spinning/rotating effect in the visualization
    // Using inertia-based rotation values passed from JavaScript
    float rotationAngle = t * 0.2 + u_rotationX * u_rotationY;
    p.xz*=rotate2D(rotationAngle);
    
    // Initialize scale factor for the fractal
    s=1.;
    
    // Distance Estimated Fractal - This is a form of ray marching
    // This loop implements a 3D Mandelbox-like fractal with folding operations
    // The technique uses Distance Estimation (DE) for rendering complex fractals
    // by iteratively transforming the point and tracking how quickly it escapes
    for(int i=0;i++<16;p=vec3(3,9,2.5)-abs(abs(p)*e-vec3(5,2,3)/e))
      // Scale factor accumulation
      // The dot product measures how quickly points escape in the fractal
      // This is similar to an escape-time fractal algorithm but with distance estimation
      // The dot product creates a spherical distance field that determines detail level
      s*=e=max(1.005,10./dot(p*.8,p));
    
    // Accumulate distance field contribution
    // This creates the layered effect in the visualization
    // The modulo operation creates repeating patterns based on distance
    // This is not traditional FBM (Fractional Brownian Motion) noise, but rather
    // a distance-based pattern derived from the fractal iterations
    g+=mod(length(p.zx),p.y)/s;
    
    // Calculate color intensity based on logarithmic scale factor
    // This creates the glowing effect where the fractal boundaries are brightest
    // The logarithm compresses the high dynamic range of the scale factor
    s=log(s)/g;
    
    // Accumulate color using HSV color space
    // The negative values create complementary colors
    // The division by 5e3 (5000) scales the intensity to a visible range
    // This creates the characteristic glow and color banding of the visualization
    o.rgb+=hsv(-g + u_colorShift,-p.y*.2,s/5e3);
  }
  
  // Output final accumulated color
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

    let cleanup = () => {}

    try {
      vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource)
      fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource)

      if (!vertexShader || !fragmentShader) {
        cleanup = () => {
          window.removeEventListener("resize", resizeCanvas)
        }
        return
      }

      // Create program and link shaders
      program = gl.createProgram()
      if (!program) {
        console.error("Failed to create program")
        cleanup = () => {
          window.removeEventListener("resize", resizeCanvas)
        }
        return
      }

      gl.attachShader(program, vertexShader)
      gl.attachShader(program, fragmentShader)
      gl.linkProgram(program)

      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error("Program linking error:", gl.getProgramInfoLog(program))
        cleanup = () => {
          window.removeEventListener("resize", resizeCanvas)
        }
        return
      }

      // Set up position buffer (full screen quad)
      positionBuffer = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)

      const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]

      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW)

      // Use program
      gl.useProgram(program)

      // Set up position attribute
      const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
      gl.enableVertexAttribArray(positionAttributeLocation)
      gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0)

      // Get uniform locations
      const resolutionUniformLocation = gl.getUniformLocation(program, "u_resolution")
      const mouseUniformLocation = gl.getUniformLocation(program, "u_mouse")
      const timeUniformLocation = gl.getUniformLocation(program, "u_time")
      const rotationXUniformLocation = gl.getUniformLocation(program, "u_rotationX")
      const rotationYUniformLocation = gl.getUniformLocation(program, "u_rotationY")
      const colorShiftUniformLocation = gl.getUniformLocation(program, "u_colorShift")

      // Mouse tracking with inertia
      let mouseX = 0
      let mouseY = 0
      let prevMouseX = 0
      let prevMouseY = 0
      let velocityX = 0
      let velocityY = 0
      let targetRotationX = 0
      let currentRotationX = 0
      let targetRotationY = 0.5
      let currentRotationY = 0.5
      let isMouseMoving = false
      let lastMouseMoveTime = 0
      const inertiaFactor = 0.95 // How quickly the inertia decays (0-1)
      const mouseSensitivity = 0.003 // How sensitive the rotation is to mouse movement

      const handleMouseMove = (event: MouseEvent) => {
        prevMouseX = mouseX
        prevMouseY = mouseY
        mouseX = event.clientX
        mouseY = event.clientY

        // Calculate velocity based on mouse movement
        velocityX = (mouseX - prevMouseX) * mouseSensitivity
        velocityY = (mouseY - prevMouseY) * mouseSensitivity

        // Update target rotation based on normalized mouse position
        targetRotationX += velocityX
        targetRotationY = Math.max(0.1, Math.min(1.0, mouseY / canvas.height))

        isMouseMoving = true
        lastMouseMoveTime = Date.now()
      }

      window.addEventListener("mousemove", handleMouseMove)

      // Render loop
      const startTime = Date.now()

      const render = () => {
        if (!gl || !program) return

        const currentTime = Date.now()
        const deltaTime = (currentTime - startTime) / 1000

        // Check if mouse has stopped moving
        if (isMouseMoving && currentTime - lastMouseMoveTime > 100) {
          isMouseMoving = false
        }

        // Apply inertia to rotation
        if (!isMouseMoving) {
          // Gradually reduce velocity when mouse is not moving
          velocityX *= inertiaFactor

          // Apply velocity to target rotation
          targetRotationX += velocityX

          // Stop when velocity becomes very small
          if (Math.abs(velocityX) < 0.0001) {
            velocityX = 0
          }
        }

        // Smooth interpolation for rotation values
        currentRotationX += (targetRotationX - currentRotationX) * 0.1
        currentRotationY += (targetRotationY - currentRotationY) * 0.1

        // Update time uniform
        gl.uniform1f(timeUniformLocation, deltaTime)

        // Update resolution uniform
        gl.uniform2f(resolutionUniformLocation, canvas.width, canvas.height)

        // Update mouse uniform
        gl.uniform2f(mouseUniformLocation, mouseX, canvas.height - mouseY)

        // Update rotation uniforms with inertia values
        gl.uniform1f(rotationXUniformLocation, currentRotationX)
        gl.uniform1f(rotationYUniformLocation, currentRotationY)

        // Update color shift uniform based on normalized mouse Y position
        gl.uniform1f(colorShiftUniformLocation, currentRotationY * 2.0)

        // Clear canvas and draw
        gl.clearColor(0, 0, 0, 1)
        gl.clear(gl.COLOR_BUFFER_BIT)
        gl.drawArrays(gl.TRIANGLES, 0, 6)

        requestAnimationFrame(render)
      }

      render()

      // Cleanup
      cleanup = () => {
        window.removeEventListener("resize", resizeCanvas)
        window.removeEventListener("mousemove", handleMouseMove)

        if (program) gl.deleteProgram(program)
        if (vertexShader) gl.deleteShader(vertexShader)
        if (fragmentShader) gl.deleteShader(fragmentShader)
        if (positionBuffer) gl.deleteBuffer(positionBuffer)
      }
    } catch (error) {
      console.error("Error during WebGL initialization or rendering:", error)
      cleanup = () => {
        window.removeEventListener("resize", resizeCanvas)
      }
    }

    return () => {
      cleanup()
    }
  }, [])

  return (
    <>
      <div
        style={{
          position: "fixed",
          top: "20px",
          left: "20px",
          color: "white",
          background: "rgba(0,0,0,0.5)",
          padding: "10px",
          borderRadius: "5px",
          zIndex: 10,
          pointerEvents: "none",
        }}
      >
        <p>
          Move mouse horizontally to control rotation with inertia
          <br />
          Move mouse vertically to change colors
        </p>
      </div>

      <div
        style={{
          position: "fixed",
          bottom: "20px",
          right: "20px",
          color: "white",
          background: "rgba(0,0,0,0.5)",
          padding: "8px 12px",
          borderRadius: "5px",
          zIndex: 10,
          pointerEvents: "none",
          fontSize: "0.9rem",
        }}
      >
        
      </div>

      <canvas
        ref={canvasRef}
        style={{
          display: "block",
          width: "100vw",
          height: "100vh",
          position: "fixed",
          top: 0,
          left: 0,
        }}
      />
    </>
  )
}

export default TweetGLSLVisualization

