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
 * Rehydration Audio-Reactive GLSL Visualization Component
 *
 * A fluid, organic audio-reactive visualization featuring flowing patterns,
 * liquid-like transformations, and hydrating color schemes that respond
 * dynamically to music with a completely different mathematical approach.
 */
const RehydrationGLSLVisualization: React.FC = () => {
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
  const audioLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const bassLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const midLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const trebleLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const frequencyDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const waveformDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  
  // Control uniform locations
  const intensityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const flowSpeedUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const colorShiftUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const waveAmplitudeUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const liquidityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const turbulenceUniformLocationRef = useRef<WebGLUniformLocation | null>(null)

  // Audio state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.7)
  const [trackName, setTrackName] = useState("")
  const [isDragOver, setIsDragOver] = useState(false)

  // Visual control state
  const [intensity, setIntensity] = useState(0.6)
  const [flowSpeed, setFlowSpeed] = useState(0.4)
  const [colorShift, setColorShift] = useState(0.3)
  const [waveAmplitude, setWaveAmplitude] = useState(0.5)
  const [liquidity, setLiquidity] = useState(0.7)
  const [turbulence, setTurbulence] = useState(0.4)
  const [smoothing, setSmoothing] = useState(0.8)

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

    const bassEnd = Math.floor(bufferLength * 0.12)
    const midEnd = Math.floor(bufferLength * 0.6)

    let bassSum = 0, midSum = 0, trebleSum = 0, totalSum = 0

    for (let i = 0; i < bassEnd; i++) {
      bassSum += dataArray[i]
    }
    bassSum /= bassEnd

    for (let i = bassEnd; i < midEnd; i++) {
      midSum += dataArray[i]
    }
    midSum /= (midEnd - bassEnd)

    for (let i = midEnd; i < bufferLength; i++) {
      trebleSum += dataArray[i]
    }
    trebleSum /= (bufferLength - midEnd)

    for (let i = 0; i < bufferLength; i++) {
      totalSum += dataArray[i]
    }
    const level = totalSum / bufferLength / 255

    const bassLevel = bassSum / 255
    const midLevel = midSum / 255
    const trebleLevel = trebleSum / 255

    for (let i = 0; i < 64; i++) {
      const index = Math.floor((i / 64) * bufferLength)
      frequencyDataRef.current[i] = dataArray[index] / 255
    }

    for (let i = 0; i < 32; i++) {
      const index = Math.floor((i / 32) * bufferLength)
      waveformDataRef.current[i] = (waveformArray[index] - 128) / 128
    }

    const smoothingFactor = smoothing
    const current = smoothedAudioDataRef.current

    current.level = current.level * smoothingFactor + level * (1 - smoothingFactor)
    current.bassLevel = current.bassLevel * smoothingFactor + bassLevel * (1 - smoothingFactor)
    current.midLevel = current.midLevel * smoothingFactor + midLevel * (1 - smoothingFactor)
    current.trebleLevel = current.trebleLevel * smoothingFactor + trebleLevel * (1 - smoothingFactor)

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

    if (audioElementRef.current) {
      audioElementRef.current.pause()
      if (audioElementRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
    }

    audioElementRef.current = audio

    if (audioContextRef.current) {
      if (sourceRef.current) {
        sourceRef.current.disconnect()
      }

      const source = audioContextRef.current.createMediaElementSource(audio)
      const analyser = audioContextRef.current.createAnalyser()

      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.25

      source.connect(analyser)
      analyser.connect(audioContextRef.current.destination)

      sourceRef.current = source
      analyserRef.current = analyser
    }
  }, [volume, initAudioContext])

  // Load audio file from URL (for default songs)
  const loadAudioFromUrl = useCallback(async (url: string, filename: string) => {
    await initAudioContext()

    const audio = new Audio()
    audio.src = url
    audio.crossOrigin = "anonymous"
    audio.volume = volume

    audio.addEventListener('loadedmetadata', () => {
      setDuration(audio.duration)
      setTrackName(filename.replace(/\.[^/.]+$/, ""))
    })

    audio.addEventListener('timeupdate', () => {
      setCurrentTime(audio.currentTime)
    })

    audio.addEventListener('ended', () => {
      setIsPlaying(false)
    })

    if (audioElementRef.current) {
      audioElementRef.current.pause()
      if (audioElementRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
    }

    audioElementRef.current = audio

    if (audioContextRef.current) {
      if (sourceRef.current) {
        sourceRef.current.disconnect()
      }

      const source = audioContextRef.current.createMediaElementSource(audio)
      const analyser = audioContextRef.current.createAnalyser()

      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.25

      source.connect(analyser)
      analyser.connect(audioContextRef.current.destination)

      sourceRef.current = source
      analyserRef.current = analyser
    }
  }, [volume, initAudioContext])

  // File upload handlers
  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('audio/')) {
      loadAudioFile(file)
    }
  }, [loadAudioFile])

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragOver(false)

    const file = event.dataTransfer.files[0]
    if (file && file.type.startsWith('audio/')) {
      loadAudioFile(file)
    }
  }, [loadAudioFile])

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false)
  }, [])

  // Audio control functions
  const togglePlayPause = useCallback(async () => {
    if (!audioElementRef.current) return

    await initAudioContext()

    if (isPlaying) {
      audioElementRef.current.pause()
      setIsPlaying(false)
    } else {
      try {
        await audioElementRef.current.play()
        setIsPlaying(true)
      } catch (error) {
        console.error('Error playing audio:', error)
      }
    }
  }, [isPlaying, initAudioContext])

  const handleSeek = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioElementRef.current) return

    const seekTime = (parseFloat(event.target.value) / 100) * duration
    audioElementRef.current.currentTime = seekTime
    setCurrentTime(seekTime)
  }, [duration])

  const handleVolumeChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(event.target.value) / 100
    setVolume(newVolume)

    if (audioElementRef.current) {
      audioElementRef.current.volume = newVolume
    }
  }, [])

  const formatTime = useCallback((time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }, [])

  // Default songs available in the app
  const defaultSongs = [
    { filename: "Charli XCX - party 4 u.mp3", displayName: "Charli XCX - Party 4 U" },
    { filename: "Benson Boone - Beautiful Things.mp3", displayName: "Benson Boone - Beautiful Things" },
    { filename: "M83 - Midnight City.mp3", displayName: "M83 - Midnight City" }
  ]

  // Load default song
  const loadDefaultSong = useCallback((filename: string, displayName: string) => {
    const url = `/songs/${encodeURIComponent(filename)}`
    loadAudioFromUrl(url, displayName)
  }, [loadAudioFromUrl])

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
      gl.viewport(0, 0, canvas.width, canvas.height)
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

    // Rehydration fragment shader with fluid, organic equations
    const fragmentShaderSource = `#version 300 es
precision highp float;

out vec4 outColor;
uniform vec2 u_resolution;
uniform float u_time;

// Audio uniforms
uniform float u_audioLevel;
uniform float u_bassLevel;
uniform float u_midLevel;
uniform float u_trebleLevel;
uniform float u_frequencyData[64];
uniform float u_waveformData[32];

// Control uniforms
uniform float u_intensity;
uniform float u_flowSpeed;
uniform float u_colorShift;
uniform float u_waveAmplitude;
uniform float u_liquidity;
uniform float u_turbulence;

// Noise functions for organic patterns
float hash21(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec2 hash22(vec2 p) {
  p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
  return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// Improved noise for fluid motion
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  
  return mix(mix(hash21(i + vec2(0,0)), hash21(i + vec2(1,0)), u.x),
             mix(hash21(i + vec2(0,1)), hash21(i + vec2(1,1)), u.x), u.y);
}

// Fluid distortion function 
vec2 fluidDistort(vec2 p, float t) {
  float audioWave = u_bassLevel * 0.8 + u_midLevel * 0.5;
  
  vec2 q = vec2(noise(p + vec2(0, t * u_flowSpeed)),
                noise(p + vec2(5.2, t * u_flowSpeed * 1.3) + audioWave));
  
  vec2 r = vec2(noise(p + 4.0 * q + vec2(1.7, 9.2) + t * u_flowSpeed * 0.7),
                noise(p + 4.0 * q + vec2(8.3, 2.8) + t * u_flowSpeed * 0.4));
  
  return p + u_liquidity * r * (1.0 + audioWave * u_intensity);
}

// Get frequency data 
float getFreq(float pos) {
  int idx = int(pos * 63.0);
  return u_frequencyData[idx];
}

// Organic color palette inspired by water and nature
vec3 hydroColor(float t, float audio) {
  t += u_colorShift + audio * 0.3;
  
  vec3 a = vec3(0.2, 0.5, 0.8);  // Deep water blue
  vec3 b = vec3(0.3, 0.8, 0.6);  // Aqua green  
  vec3 c = vec3(0.6, 0.9, 1.2);  // Light cyan
  vec3 d = vec3(0.1, 0.4, 0.7);  // Dark blue base
  
  return a + b * cos(6.28318 * (c * t + d));
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
  float t = u_time;
  vec3 col = vec3(0);
  
  // Audio-reactive scaling and movement
  float audioScale = 1.0 + (u_audioLevel * 0.4 + u_bassLevel * 0.3) * u_intensity;
  uv *= audioScale;
  
  // Multi-layered fluid distortion
  for (int i = 0; i < 4; i++) {
    float layer = float(i);
    vec2 p = uv * (2.0 + layer * 0.5);
    
    // Apply fluid distortion with audio influence
    p = fluidDistort(p, t + layer * 0.3);
    
    // Create flowing patterns with frequency data
    float freqInfluence = getFreq(layer * 0.25) * u_waveAmplitude;
    
    // Organic wave equation with audio reactivity
    float wave1 = sin(p.x * 3.0 + t * u_flowSpeed * 2.0 + freqInfluence * 3.0);
    float wave2 = cos(p.y * 2.5 + t * u_flowSpeed * 1.7 + u_trebleLevel * 2.0);
    float wave3 = sin(length(p) * 4.0 - t * u_flowSpeed * 2.5 + u_midLevel * 4.0);
    
    // Combine waves with turbulence
    float pattern = (wave1 + wave2 + wave3) * 0.33;
    pattern += noise(p * 6.0 + t * u_flowSpeed) * u_turbulence * 0.5;
    
    // Audio-reactive intensity modulation
    pattern *= 1.0 + (u_bassLevel * 0.3 + getFreq(0.1 + layer * 0.2) * 0.4) * u_intensity;
    
    // Create flowing color with audio influence
    float audioColorShift = u_audioLevel * 0.2 + u_trebleLevel * 0.15;
    vec3 layerColor = hydroColor(pattern * 0.5 + layer * 0.3 + t * 0.1, audioColorShift);
    
    // Blend layers with audio-reactive opacity
    float opacity = 0.3 + getFreq(layer * 0.3) * 0.2 * u_intensity;
    col = mix(col, layerColor, opacity * (1.0 - layer * 0.2));
  }
  
  // Final audio enhancement and glow
  col *= 1.0 + u_audioLevel * 0.3 * u_intensity;
  
  // Add subtle waveform visualization
  vec2 wavePos = gl_FragCoord.xy / u_resolution.xy;
  if (wavePos.y < 0.1 && wavePos.x < 0.8) {
    int waveIdx = int(wavePos.x * 31.0);
    float waveVal = u_waveformData[waveIdx];
    col += vec3(0.1, 0.3, 0.5) * abs(waveVal) * 3.0 * u_intensity;
  }
  
  // Soft vignette for focus
  float vignette = 1.0 - length(uv * 0.5);
  col *= 0.7 + 0.3 * vignette;
  
  outColor = vec4(col, 1.0);
}
`

    const createShader = (gl: WebGL2RenderingContext, type: number, source: string) => {
      const shader = gl.createShader(type)
      if (!shader) return null

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
    if (!program) return

    gl.attachShader(program, vertexShader)
    gl.attachShader(program, fragmentShader)
    gl.linkProgram(program)

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error("Program linking error:", gl.getProgramInfoLog(program))
      return
    }

    programRef.current = program

    const positionBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
    const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW)

    const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
    gl.enableVertexAttribArray(positionAttributeLocation)
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0)

    // Get uniform locations
    resolutionUniformLocationRef.current = gl.getUniformLocation(program, "u_resolution")
    timeUniformLocationRef.current = gl.getUniformLocation(program, "u_time")
    audioLevelUniformLocationRef.current = gl.getUniformLocation(program, "u_audioLevel")
    bassLevelUniformLocationRef.current = gl.getUniformLocation(program, "u_bassLevel")
    midLevelUniformLocationRef.current = gl.getUniformLocation(program, "u_midLevel")
    trebleLevelUniformLocationRef.current = gl.getUniformLocation(program, "u_trebleLevel")
    frequencyDataUniformLocationRef.current = gl.getUniformLocation(program, "u_frequencyData")
    waveformDataUniformLocationRef.current = gl.getUniformLocation(program, "u_waveformData")
    
    intensityUniformLocationRef.current = gl.getUniformLocation(program, "u_intensity")
    flowSpeedUniformLocationRef.current = gl.getUniformLocation(program, "u_flowSpeed")
    colorShiftUniformLocationRef.current = gl.getUniformLocation(program, "u_colorShift")
    waveAmplitudeUniformLocationRef.current = gl.getUniformLocation(program, "u_waveAmplitude")
    liquidityUniformLocationRef.current = gl.getUniformLocation(program, "u_liquidity")
    turbulenceUniformLocationRef.current = gl.getUniformLocation(program, "u_turbulence")

    const startTime = performance.now()

    const render = () => {
      if (!programRef.current || !gl) return
      gl.useProgram(programRef.current)

      const audioData = analyzeAudio()
      const currentTime = (performance.now() - startTime) / 1000

      // Update uniforms
      if (timeUniformLocationRef.current) {
        gl.uniform1f(timeUniformLocationRef.current, currentTime)
      }
      if (resolutionUniformLocationRef.current) {
        gl.uniform2f(resolutionUniformLocationRef.current, canvas.width, canvas.height)
      }

      // Audio uniforms
      if (audioLevelUniformLocationRef.current) {
        gl.uniform1f(audioLevelUniformLocationRef.current, audioData.level)
      }
      if (bassLevelUniformLocationRef.current) {
        gl.uniform1f(bassLevelUniformLocationRef.current, audioData.bassLevel)
      }
      if (midLevelUniformLocationRef.current) {
        gl.uniform1f(midLevelUniformLocationRef.current, audioData.midLevel)
      }
      if (trebleLevelUniformLocationRef.current) {
        gl.uniform1f(trebleLevelUniformLocationRef.current, audioData.trebleLevel)
      }
      if (frequencyDataUniformLocationRef.current) {
        gl.uniform1fv(frequencyDataUniformLocationRef.current, audioData.frequencyData)
      }
      if (waveformDataUniformLocationRef.current) {
        gl.uniform1fv(waveformDataUniformLocationRef.current, audioData.waveformData)
      }

      // Control uniforms
      if (intensityUniformLocationRef.current) {
        gl.uniform1f(intensityUniformLocationRef.current, intensity)
      }
      if (flowSpeedUniformLocationRef.current) {
        gl.uniform1f(flowSpeedUniformLocationRef.current, flowSpeed)
      }
      if (colorShiftUniformLocationRef.current) {
        gl.uniform1f(colorShiftUniformLocationRef.current, colorShift)
      }
      if (waveAmplitudeUniformLocationRef.current) {
        gl.uniform1f(waveAmplitudeUniformLocationRef.current, waveAmplitude)
      }
      if (liquidityUniformLocationRef.current) {
        gl.uniform1f(liquidityUniformLocationRef.current, liquidity)
      }
      if (turbulenceUniformLocationRef.current) {
        gl.uniform1f(turbulenceUniformLocationRef.current, turbulence)
      }

      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.drawArrays(gl.TRIANGLES, 0, 6)

      animationRef.current = requestAnimationFrame(render)
    }

    render()

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      if (program) {
        gl.deleteProgram(program)
      }
      gl.deleteShader(vertexShader)
      gl.deleteShader(fragmentShader)
      gl.deleteBuffer(positionBuffer)
    }
  }, [analyzeAudio, intensity, flowSpeed, colorShift, waveAmplitude, liquidity, turbulence])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioElementRef.current) {
        audioElementRef.current.pause()
        if (audioElementRef.current.src.startsWith('blob:')) {
          URL.revokeObjectURL(audioElementRef.current.src)
        }
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
    }
  }, [])

  return (
    <div
      style={{ position: "relative", width: "100vw", height: "100vh" }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <canvas
        ref={canvasRef}
        style={{
          display: "block",
          width: "100%",
          height: "100%",
          position: "absolute",
          top: 0,
          left: 0,
        }}
      />

      {/* Audio Controls with hydration theme */}
      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          background: "rgba(0, 20, 40, 0.85)",
          borderRadius: "12px",
          padding: "18px",
          color: "#E0F7FF",
          fontFamily: "sans-serif",
          fontSize: "14px",
          minWidth: "320px",
          zIndex: 10,
          backdropFilter: "blur(12px)",
          border: "1px solid rgba(100, 200, 255, 0.2)",
          boxShadow: "0 8px 32px rgba(0, 100, 200, 0.1)"
        }}
      >
        <div style={{ marginBottom: "12px", fontSize: "16px", fontWeight: "bold", color: "#7DD3FC" }}>
          🌊 Rehydration
        </div>

        {/* File Upload */}
        <div style={{ marginBottom: "15px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontSize: "12px", opacity: 0.8 }}>
            Upload Audio File
          </label>
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            style={{
              width: "100%",
              padding: "8px",
              background: "rgba(100, 200, 255, 0.1)",
              border: "1px solid rgba(100, 200, 255, 0.3)",
              borderRadius: "6px",
              color: "#E0F7FF",
              fontSize: "12px"
            }}
          />
        </div>

        {/* Default Songs */}
        <div style={{ marginBottom: "15px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontSize: "12px", opacity: 0.8 }}>
            Or Choose a Default Song
          </label>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            {defaultSongs.map((song, index) => (
              <button
                key={index}
                onClick={() => loadDefaultSong(song.filename, song.displayName)}
                style={{
                  width: "100%",
                  padding: "8px",
                  background: "rgba(100, 200, 255, 0.1)",
                  border: "1px solid rgba(100, 200, 255, 0.3)",
                  borderRadius: "6px",
                  color: "#E0F7FF",
                  fontSize: "12px",
                  textAlign: "left",
                  cursor: "pointer",
                  transition: "all 0.2s"
                }}
                onMouseEnter={(e) => {
                  (e.target as HTMLButtonElement).style.background = "rgba(100, 200, 255, 0.2)"
                  ;(e.target as HTMLButtonElement).style.borderColor = "rgba(100, 200, 255, 0.5)"
                }}
                onMouseLeave={(e) => {
                  (e.target as HTMLButtonElement).style.background = "rgba(100, 200, 255, 0.1)"
                  ;(e.target as HTMLButtonElement).style.borderColor = "rgba(100, 200, 255, 0.3)"
                }}
              >
                {song.displayName}
              </button>
            ))}
          </div>
        </div>

        {/* Track Info */}
        {trackName && (
          <div style={{ marginBottom: "15px", fontSize: "13px", fontWeight: "bold", color: "#7DD3FC" }}>
            {trackName}
          </div>
        )}

        {/* Play/Pause Button */}
        <div style={{ marginBottom: "15px", textAlign: "center" }}>
          <button
            onClick={togglePlayPause}
            disabled={!audioElementRef.current}
            style={{
              padding: "12px 24px",
              background: isPlaying ? "linear-gradient(45deg, #FF6B6B, #FF8E8E)" : "linear-gradient(45deg, #4ECDC4, #7DD3FC)",
              border: "none",
              borderRadius: "25px",
              color: "white",
              fontWeight: "bold",
              cursor: "pointer",
              fontSize: "14px",
              opacity: !audioElementRef.current ? 0.5 : 1,
              boxShadow: "0 4px 15px rgba(100, 200, 255, 0.3)"
            }}
          >
            {isPlaying ? "⏸ Pause" : "▶ Play"}
          </button>
        </div>

        {/* Timeline */}
        {duration > 0 && (
          <div style={{ marginBottom: "15px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "5px" }}>
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={(currentTime / duration) * 100}
              onChange={handleSeek}
              style={{
                width: "100%",
                height: "4px",
                background: "rgba(100, 200, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>
        )}

        {/* Volume Control */}
        <div style={{ marginBottom: "15px" }}>
          <label style={{ display: "block", marginBottom: "5px", fontSize: "12px", opacity: 0.8 }}>
            Volume: {Math.round(volume * 100)}%
          </label>
          <input
            type="range"
            min="0"
            max="100"
            value={volume * 100}
            onChange={handleVolumeChange}
            style={{
              width: "100%",
              height: "4px",
              background: "rgba(100, 200, 255, 0.3)",
              outline: "none",
              borderRadius: "2px"
            }}
          />
        </div>

        {/* Visual Controls */}
        <div style={{ marginBottom: "15px", borderTop: "1px solid rgba(100, 200, 255, 0.2)", paddingTop: "15px" }}>
          <div style={{ marginBottom: "8px", fontSize: "13px", fontWeight: "bold", opacity: 0.9 }}>
            Fluid Controls
          </div>

          {/* Intensity */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Intensity: {Math.round(intensity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={intensity * 100}
              onChange={(e) => setIntensity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(100, 200, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Flow Speed */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Flow Speed: {Math.round(flowSpeed * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={flowSpeed * 100}
              onChange={(e) => setFlowSpeed(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(100, 200, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Color Shift */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Color Shift: {Math.round(colorShift * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={colorShift * 100}
              onChange={(e) => setColorShift(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(100, 200, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Wave Amplitude */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Wave Amplitude: {Math.round(waveAmplitude * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="150"
              value={waveAmplitude * 100}
              onChange={(e) => setWaveAmplitude(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(100, 200, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Liquidity */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Liquidity: {Math.round(liquidity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="150"
              value={liquidity * 100}
              onChange={(e) => setLiquidity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(100, 200, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Turbulence */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Turbulence: {Math.round(turbulence * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={turbulence * 100}
              onChange={(e) => setTurbulence(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(100, 200, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Smoothing */}
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Smoothing: {Math.round(smoothing * 100)}%
            </label>
            <input
              type="range"
              min="10"
              max="95"
              value={smoothing * 100}
              onChange={(e) => setSmoothing(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(100, 200, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          {/* Reset Button */}
          <div style={{ textAlign: "center", marginTop: "10px" }}>
            <button
              onClick={() => {
                setIntensity(0.6)
                setFlowSpeed(0.4)
                setColorShift(0.3)
                setWaveAmplitude(0.5)
                setLiquidity(0.7)
                setTurbulence(0.4)
                setSmoothing(0.8)
              }}
              style={{
                padding: "8px 16px",
                background: "rgba(100, 200, 255, 0.2)",
                border: "1px solid rgba(100, 200, 255, 0.4)",
                borderRadius: "6px",
                color: "#E0F7FF",
                fontSize: "11px",
                cursor: "pointer"
              }}
            >
              Reset Fluid Controls
            </button>
          </div>
        </div>

        {/* Audio Levels Display */}
        {isPlaying && (
          <div style={{ fontSize: "10px", opacity: 0.7, color: "#7DD3FC" }}>
            <div>Level: {Math.round(smoothedAudioDataRef.current.level * 100)}%</div>
            <div>Bass: {Math.round(smoothedAudioDataRef.current.bassLevel * 100)}%</div>
            <div>Mid: {Math.round(smoothedAudioDataRef.current.midLevel * 100)}%</div>
            <div>Treble: {Math.round(smoothedAudioDataRef.current.trebleLevel * 100)}%</div>
          </div>
        )}
      </div>

      {/* Drag and Drop Overlay */}
      {isDragOver && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            background: "rgba(100, 200, 255, 0.15)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 20,
            fontSize: "24px",
            color: "#7DD3FC",
            fontFamily: "sans-serif",
            fontWeight: "bold"
          }}
        >
          🌊 Drop audio file to rehydrate
        </div>
      )}
    </div>
  )
}

export default RehydrationGLSLVisualization 