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
 * Waveform GLSL Visualization Component
 *
 * A React component that renders an audio-reactive waveform visualization
 * with real-time waveform data display and customizable visual controls.
 */
const WaveformGLSL: React.FC = () => {
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
  
  // Audio uniform locations
  const audioLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const bassLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const midLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const trebleLevelUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const waveformDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const frequencyDataUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  
  // Visual control uniform locations
  const audioIntensityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const waveHeightUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const glowIntensityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const colorShiftUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const smoothnessUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const reflectionStrengthUniformLocationRef = useRef<WebGLUniformLocation | null>(null)

  // Audio state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.7)
  const [trackName, setTrackName] = useState("")
  const [isDragOver, setIsDragOver] = useState(false)

  // Visual controls
  const [audioIntensity, setAudioIntensity] = useState(0.8)
  const [waveHeight, setWaveHeight] = useState(2.0)
  const [glowIntensity, setGlowIntensity] = useState(1.5)
  const [colorShift, setColorShift] = useState(0.0)
  const [smoothness, setSmoothness] = useState(0.5)
  const [reflectionStrength, setReflectionStrength] = useState(0.6)

  // Audio data with increased resolution for waveform
  const smoothedAudioDataRef = useRef<AudioData>({
    level: 0,
    bassLevel: 0,
    midLevel: 0,
    trebleLevel: 0,
    frequencyData: new Float32Array(256),
    waveformData: new Float32Array(256)
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

  // Enhanced audio analysis function for waveform
  const analyzeAudio = useCallback(() => {
    const analyser = analyserRef.current
    if (!analyser) return smoothedAudioDataRef.current

    const bufferLength = analyser.frequencyBinCount
    const freqDataArray = new Uint8Array(bufferLength)
    const waveDataArray = new Uint8Array(analyser.fftSize)
    
    // Get frequency data
    analyser.getByteFrequencyData(freqDataArray)
    // Get waveform data
    analyser.getByteTimeDomainData(waveDataArray)

    // Calculate frequency bands
    const bassEnd = Math.floor(bufferLength * 0.1)
    const midEnd = Math.floor(bufferLength * 0.5)

    let bassSum = 0, midSum = 0, trebleSum = 0, totalSum = 0

    for (let i = 0; i < bassEnd; i++) {
      bassSum += freqDataArray[i]
    }
    bassSum /= bassEnd

    for (let i = bassEnd; i < midEnd; i++) {
      midSum += freqDataArray[i]
    }
    midSum /= (midEnd - bassEnd)

    for (let i = midEnd; i < bufferLength; i++) {
      trebleSum += freqDataArray[i]
    }
    trebleSum /= (bufferLength - midEnd)

    for (let i = 0; i < bufferLength; i++) {
      totalSum += freqDataArray[i]
    }
    const level = totalSum / bufferLength / 255

    // Smooth the audio data
    const smoothingFactor = 0.85
    const current = smoothedAudioDataRef.current

    current.level = current.level * smoothingFactor + level * (1 - smoothingFactor)
    current.bassLevel = current.bassLevel * smoothingFactor + (bassSum / 255) * (1 - smoothingFactor)
    current.midLevel = current.midLevel * smoothingFactor + (midSum / 255) * (1 - smoothingFactor)
    current.trebleLevel = current.trebleLevel * smoothingFactor + (trebleSum / 255) * (1 - smoothingFactor)

    // Update frequency data with less smoothing for responsiveness
    const freqSmoothing = 0.3
    for (let i = 0; i < Math.min(256, bufferLength); i++) {
      current.frequencyData[i] = current.frequencyData[i] * freqSmoothing + 
                                 (freqDataArray[i] / 255) * (1 - freqSmoothing)
    }

    // Update waveform data - no smoothing for maximum responsiveness
    for (let i = 0; i < Math.min(256, waveDataArray.length); i++) {
      current.waveformData[i] = (waveDataArray[i] - 128) / 128 // Normalize to -1 to 1
    }

    return current
  }, [])

  // Load audio file
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

      // Higher FFT size for better waveform resolution
      analyser.fftSize = 2048
      analyser.smoothingTimeConstant = 0.1 // Less smoothing for better responsiveness

      source.connect(analyser)
      analyser.connect(audioContextRef.current.destination)

      sourceRef.current = source
      analyserRef.current = analyser
    }
  }, [volume, initAudioContext])

  // Load audio from URL
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

      analyser.fftSize = 2048
      analyser.smoothingTimeConstant = 0.1

      source.connect(analyser)
      analyser.connect(audioContextRef.current.destination)

      sourceRef.current = source
      analyserRef.current = analyser
    }
  }, [volume, initAudioContext])

  // File handlers
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

  // Audio controls
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

  // Default songs
  const defaultSongs = [
    { filename: "Charli XCX - party 4 u.mp3", displayName: "Charli XCX - Party 4 U" },
    { filename: "Benson Boone - Beautiful Things.mp3", displayName: "Benson Boone - Beautiful Things" },
    { filename: "M83 - Midnight City.mp3", displayName: "M83 - Midnight City" }
  ]

  const loadDefaultSong = useCallback((filename: string, displayName: string) => {
    const url = `/songs/${encodeURIComponent(filename)}`
    loadAudioFromUrl(url, displayName)
  }, [loadAudioFromUrl])

  // WebGL setup and render loop
  useEffect(() => {
    let program: WebGLProgram | null = null
    let vertexShader: WebGLShader | null = null
    let fragmentShader: WebGLShader | null = null
    let positionBuffer: WebGLBuffer | null = null
    let gl: WebGL2RenderingContext | null = null
    const canvas = canvasRef.current
    if (!canvas) return

    gl = canvas.getContext("webgl2")
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

    const fragmentShaderSource = `#version 300 es
precision highp float;

out vec4 fragColor;
uniform vec2 iResolution;
uniform float iTime;

// Audio uniforms
uniform float u_audioLevel;
uniform float u_bassLevel;
uniform float u_midLevel;
uniform float u_trebleLevel;
uniform float u_waveformData[256];
uniform float u_frequencyData[256];

// Visual control uniforms
uniform float u_audioIntensity;
uniform float u_waveHeight;
uniform float u_glowIntensity;
uniform float u_colorShift;
uniform float u_smoothness;
uniform float u_reflectionStrength;

// Get waveform value at position
float getWaveform(float x) {
    // Map x position to waveform array index
    float index = (x * 0.5 + 0.5) * 255.0;
    int i0 = int(floor(index));
    int i1 = int(ceil(index));
    float fract = fract(index);
    
    // Interpolate between samples for smoothness
    if (i0 >= 0 && i1 < 256) {
        float w0 = u_waveformData[i0];
        float w1 = u_waveformData[i1];
        return mix(w0, w1, fract * u_smoothness);
    }
    return 0.0;
}

// Get frequency spectrum value
float getFrequency(float x) {
    float index = abs(x) * 128.0;
    int i = int(min(index, 255.0));
    if (i >= 0 && i < 256) {
        return u_frequencyData[i];
    }
    return 0.0;
}

void main() {
    vec2 I = gl_FragCoord.xy;
    
    // Raymarch variables
    float i, d, z, r;
    vec4 O = vec4(0.0);
    
    // Raymarch loop
    for(O *= i; i++ < 90.0;
        O += (cos(z * 0.5 + iTime + vec4(0, 2, 4, 3) + u_colorShift) + 1.3) / d / z * u_glowIntensity)
    {
        // Raymarch sample point
        vec3 p = z * normalize(vec3(I + I, 0) - iResolution.xyy);
        
        // Shift camera and get reflection coordinates
        r = max(-++p, 0.).y * u_reflectionStrength;
        // Mirror
        p.y += r + r;
        
        // Get waveform data instead of sine waves
        float waveValue = getWaveform(p.x / 20.0);
        float freqValue = getFrequency(p.x / 20.0);
        
        // Combine waveform and frequency data
        float audioHeight = u_waveHeight * u_audioIntensity;
        p.y += waveValue * audioHeight * (1.0 + u_bassLevel * 2.0);
        p.y += freqValue * audioHeight * 0.5 * (1.0 + u_trebleLevel);
        
        // Add some movement based on audio
        for(d = 1.; d < 30.; d += d) {
            float audioMod = 1.0 + u_midLevel * u_audioIntensity;
            p.y += cos(p * d + 2. * iTime * cos(d) * audioMod + z).x / d * 0.3;
        }
        
        // Step forward (reflections are softer)
        z += d = (0.1 * r + abs(p.y - 1.) / (1. + r + r + r * r) + max(d = p.z + 3., -d * 0.1)) / 8.;
    }
    
    // Apply audio-reactive brightness
    O *= 1.0 + u_audioLevel * 0.5 * u_audioIntensity;
    
    // Tanh tonemapping
    fragColor = tanh(O / 900.0);
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

      positionBuffer = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
      const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW)

      gl.useProgram(program)

      const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
      gl.enableVertexAttribArray(positionAttributeLocation)
      gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0)

      // Get uniform locations
      const resolutionUniformLocation = gl.getUniformLocation(program, "iResolution")
      const timeUniformLocation = gl.getUniformLocation(program, "iTime")
      
      const audioLevelUniformLocation = gl.getUniformLocation(program, "u_audioLevel")
      const bassLevelUniformLocation = gl.getUniformLocation(program, "u_bassLevel")
      const midLevelUniformLocation = gl.getUniformLocation(program, "u_midLevel")
      const trebleLevelUniformLocation = gl.getUniformLocation(program, "u_trebleLevel")
      const waveformDataUniformLocation = gl.getUniformLocation(program, "u_waveformData")
      const frequencyDataUniformLocation = gl.getUniformLocation(program, "u_frequencyData")
      
      const audioIntensityUniformLocation = gl.getUniformLocation(program, "u_audioIntensity")
      const waveHeightUniformLocation = gl.getUniformLocation(program, "u_waveHeight")
      const glowIntensityUniformLocation = gl.getUniformLocation(program, "u_glowIntensity")
      const colorShiftUniformLocation = gl.getUniformLocation(program, "u_colorShift")
      const smoothnessUniformLocation = gl.getUniformLocation(program, "u_smoothness")
      const reflectionStrengthUniformLocation = gl.getUniformLocation(program, "u_reflectionStrength")

      const startTime = Date.now()

      const render = () => {
        if (!gl || !program) return

        const currentTime = Date.now()
        const deltaTime = (currentTime - startTime) / 1000

        const audioData = analyzeAudio()

        gl.uniform1f(timeUniformLocation, deltaTime)
        gl.uniform2f(resolutionUniformLocation, canvas.width, canvas.height)

        if (audioLevelUniformLocation) {
          gl.uniform1f(audioLevelUniformLocation, audioData.level)
        }
        if (bassLevelUniformLocation) {
          gl.uniform1f(bassLevelUniformLocation, audioData.bassLevel)
        }
        if (midLevelUniformLocation) {
          gl.uniform1f(midLevelUniformLocation, audioData.midLevel)
        }
        if (trebleLevelUniformLocation) {
          gl.uniform1f(trebleLevelUniformLocation, audioData.trebleLevel)
        }
        
        // Pass waveform data
        if (waveformDataUniformLocation) {
          gl.uniform1fv(waveformDataUniformLocation, audioData.waveformData)
        }
        
        // Pass frequency data
        if (frequencyDataUniformLocation) {
          gl.uniform1fv(frequencyDataUniformLocation, audioData.frequencyData)
        }

        if (audioIntensityUniformLocation) {
          gl.uniform1f(audioIntensityUniformLocation, audioIntensity)
        }
        if (waveHeightUniformLocation) {
          gl.uniform1f(waveHeightUniformLocation, waveHeight)
        }
        if (glowIntensityUniformLocation) {
          gl.uniform1f(glowIntensityUniformLocation, glowIntensity)
        }
        if (colorShiftUniformLocation) {
          gl.uniform1f(colorShiftUniformLocation, colorShift)
        }
        if (smoothnessUniformLocation) {
          gl.uniform1f(smoothnessUniformLocation, smoothness)
        }
        if (reflectionStrengthUniformLocation) {
          gl.uniform1f(reflectionStrengthUniformLocation, reflectionStrength)
        }

        gl.clearColor(0, 0, 0, 1)
        gl.clear(gl.COLOR_BUFFER_BIT)
        gl.drawArrays(gl.TRIANGLES, 0, 6)

        requestAnimationFrame(render)
      }

      render()

      cleanup = () => {
        window.removeEventListener("resize", resizeCanvas)
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
  }, [analyzeAudio, audioIntensity, waveHeight, glowIntensity, colorShift, smoothness, reflectionStrength])

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

      {/* Audio Controls Panel */}
      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          background: "rgba(0, 0, 0, 0.85)",
          borderRadius: "12px",
          padding: "20px",
          color: "white",
          fontFamily: "system-ui, -apple-system, sans-serif",
          fontSize: "14px",
          minWidth: "320px",
          zIndex: 10,
          backdropFilter: "blur(20px)",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)"
        }}
      >
        <h3 style={{ margin: "0 0 15px 0", fontSize: "16px", fontWeight: "600", opacity: 0.9 }}>
          Waveform Audio Visualizer
        </h3>

        {/* File Upload */}
        <div style={{ marginBottom: "20px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontSize: "12px", opacity: 0.7 }}>
            Upload Audio File
          </label>
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            style={{
              width: "100%",
              padding: "10px",
              background: "rgba(255, 255, 255, 0.08)",
              border: "1px solid rgba(255, 255, 255, 0.2)",
              borderRadius: "6px",
              color: "white",
              fontSize: "12px",
              cursor: "pointer"
            }}
          />
        </div>

        {/* Default Songs */}
        <div style={{ marginBottom: "20px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontSize: "12px", opacity: 0.7 }}>
            Sample Tracks
          </label>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {defaultSongs.map((song, index) => (
              <button
                key={index}
                onClick={() => loadDefaultSong(song.filename, song.displayName)}
                style={{
                  padding: "8px 12px",
                  background: "rgba(255, 255, 255, 0.08)",
                  border: "1px solid rgba(255, 255, 255, 0.2)",
                  borderRadius: "6px",
                  color: "white",
                  fontSize: "12px",
                  textAlign: "left",
                  cursor: "pointer",
                  transition: "all 0.2s"
                }}
                onMouseEnter={(e) => {
                  (e.target as HTMLButtonElement).style.background = "rgba(255, 255, 255, 0.15)"
                }}
                onMouseLeave={(e) => {
                  (e.target as HTMLButtonElement).style.background = "rgba(255, 255, 255, 0.08)"
                }}
              >
                {song.displayName}
              </button>
            ))}
          </div>
        </div>

        {/* Track Info */}
        {trackName && (
          <div style={{ marginBottom: "15px", fontSize: "13px", fontWeight: "500" }}>
            {trackName}
          </div>
        )}

        {/* Play/Pause Button */}
        <div style={{ marginBottom: "20px", textAlign: "center" }}>
          <button
            onClick={togglePlayPause}
            disabled={!audioElementRef.current}
            style={{
              padding: "12px 32px",
              background: isPlaying 
                ? "linear-gradient(135deg, #ff4458 0%, #ff6b6b 100%)" 
                : "linear-gradient(135deg, #4ecdc4 0%, #44a3aa 100%)",
              border: "none",
              borderRadius: "24px",
              color: "white",
              fontWeight: "600",
              cursor: "pointer",
              fontSize: "14px",
              opacity: !audioElementRef.current ? 0.5 : 1,
              transition: "all 0.3s",
              boxShadow: "0 4px 15px rgba(0, 0, 0, 0.3)"
            }}
          >
            {isPlaying ? "⏸ Pause" : "▶ Play"}
          </button>
        </div>

        {/* Timeline */}
        {duration > 0 && (
          <div style={{ marginBottom: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "8px", opacity: 0.7 }}>
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
                height: "6px",
                background: "rgba(255, 255, 255, 0.2)",
                outline: "none",
                borderRadius: "3px",
                cursor: "pointer"
              }}
            />
          </div>
        )}

        {/* Volume Control */}
        <div style={{ marginBottom: "20px" }}>
          <label style={{ display: "block", marginBottom: "8px", fontSize: "12px", opacity: 0.7 }}>
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
              height: "6px",
              background: "rgba(255, 255, 255, 0.2)",
              outline: "none",
              borderRadius: "3px",
              cursor: "pointer"
            }}
          />
        </div>

        {/* Visual Controls */}
        <div style={{ 
          borderTop: "1px solid rgba(255, 255, 255, 0.15)", 
          paddingTop: "20px",
          marginTop: "20px"
        }}>
          <h4 style={{ margin: "0 0 15px 0", fontSize: "14px", fontWeight: "500", opacity: 0.8 }}>
            Visual Controls
          </h4>

          {/* Audio Intensity */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Audio Reactivity: {Math.round(audioIntensity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={audioIntensity * 100}
              onChange={(e) => setAudioIntensity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "4px",
                background: "rgba(255, 255, 255, 0.2)",
                outline: "none",
                borderRadius: "2px",
                cursor: "pointer"
              }}
            />
          </div>

          {/* Wave Height */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Wave Height: {waveHeight.toFixed(1)}
            </label>
            <input
              type="range"
              min="50"
              max="500"
              value={waveHeight * 100}
              onChange={(e) => setWaveHeight(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "4px",
                background: "rgba(255, 255, 255, 0.2)",
                outline: "none",
                borderRadius: "2px",
                cursor: "pointer"
              }}
            />
          </div>

          {/* Glow Intensity */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Glow Intensity: {glowIntensity.toFixed(1)}
            </label>
            <input
              type="range"
              min="50"
              max="300"
              value={glowIntensity * 100}
              onChange={(e) => setGlowIntensity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "4px",
                background: "rgba(255, 255, 255, 0.2)",
                outline: "none",
                borderRadius: "2px",
                cursor: "pointer"
              }}
            />
          </div>

          {/* Color Shift */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Color Shift
            </label>
            <input
              type="range"
              min="-314"
              max="314"
              value={colorShift * 100}
              onChange={(e) => setColorShift(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "4px",
                background: "rgba(255, 255, 255, 0.2)",
                outline: "none",
                borderRadius: "2px",
                cursor: "pointer"
              }}
            />
          </div>

          {/* Smoothness */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Wave Smoothness: {Math.round(smoothness * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={smoothness * 100}
              onChange={(e) => setSmoothness(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "4px",
                background: "rgba(255, 255, 255, 0.2)",
                outline: "none",
                borderRadius: "2px",
                cursor: "pointer"
              }}
            />
          </div>

          {/* Reflection Strength */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Reflection: {Math.round(reflectionStrength * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={reflectionStrength * 100}
              onChange={(e) => setReflectionStrength(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "4px",
                background: "rgba(255, 255, 255, 0.2)",
                outline: "none",
                borderRadius: "2px",
                cursor: "pointer"
              }}
            />
          </div>

          {/* Reset Button */}
          <div style={{ textAlign: "center", marginTop: "15px" }}>
            <button
              onClick={() => {
                setAudioIntensity(0.8)
                setWaveHeight(2.0)
                setGlowIntensity(1.5)
                setColorShift(0.0)
                setSmoothness(0.5)
                setReflectionStrength(0.6)
              }}
              style={{
                padding: "8px 16px",
                background: "rgba(255, 255, 255, 0.1)",
                border: "1px solid rgba(255, 255, 255, 0.2)",
                borderRadius: "6px",
                color: "white",
                fontSize: "11px",
                cursor: "pointer",
                transition: "all 0.2s"
              }}
              onMouseEnter={(e) => {
                (e.target as HTMLButtonElement).style.background = "rgba(255, 255, 255, 0.15)"
              }}
              onMouseLeave={(e) => {
                (e.target as HTMLButtonElement).style.background = "rgba(255, 255, 255, 0.1)"
              }}
            >
              Reset to Defaults
            </button>
          </div>
        </div>

        {/* Audio Levels Display */}
        {isPlaying && (
          <div style={{ 
            marginTop: "15px", 
            paddingTop: "15px", 
            borderTop: "1px solid rgba(255, 255, 255, 0.15)",
            fontSize: "10px", 
            opacity: 0.6 
          }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px" }}>
              <div>Level: {Math.round(smoothedAudioDataRef.current.level * 100)}%</div>
              <div>Bass: {Math.round(smoothedAudioDataRef.current.bassLevel * 100)}%</div>
              <div>Mid: {Math.round(smoothedAudioDataRef.current.midLevel * 100)}%</div>
              <div>Treble: {Math.round(smoothedAudioDataRef.current.trebleLevel * 100)}%</div>
            </div>
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
            background: "rgba(78, 205, 196, 0.1)",
            backdropFilter: "blur(10px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 20,
            fontSize: "24px",
            color: "white",
            fontFamily: "system-ui, -apple-system, sans-serif",
            fontWeight: "600"
          }}
        >
          Drop audio file to visualize
        </div>
      )}
    </div>
  )
}

export default WaveformGLSL 