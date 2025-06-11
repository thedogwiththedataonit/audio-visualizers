"use client"

import { useRef, useEffect, useState, useCallback } from "react"

interface AudioData {
  level: number
  bassLevel: number
  midLevel: number
  trebleLevel: number
  frequencyData: Float32Array
  waveformData: Float32Array
}

export default function CrystalGLSLVisualization() {
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
  const rotationSpeedUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const colorSensitivityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const beatPulseUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const fractalComplexityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const scaleReactivityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)

  // Audio state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.7)
  const [trackName, setTrackName] = useState("")
  const [isDragOver, setIsDragOver] = useState(false)

  // Visual control state
  const [intensity, setIntensity] = useState(0.5)
  const [rotationSpeed, setRotationSpeed] = useState(0.5)
  const [colorSensitivity, setColorSensitivity] = useState(0.5)
  const [beatPulse, setBeatPulse] = useState(0.5)
  const [fractalComplexity, setFractalComplexity] = useState(0.5)
  const [smoothing, setSmoothing] = useState(0.85)
  const [scaleReactivity, setScaleReactivity] = useState(0.5)

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

    const bassEnd = Math.floor(bufferLength * 0.1)
    const midEnd = Math.floor(bufferLength * 0.5)

    let bassSum = 0, midSum = 0, trebleSum = 0, totalSum = 0

    for (let i = 0; i < bassEnd; i++) bassSum += dataArray[i]
    bassSum /= bassEnd
    for (let i = bassEnd; i < midEnd; i++) midSum += dataArray[i]
    midSum /= (midEnd - bassEnd)
    for (let i = midEnd; i < bufferLength; i++) trebleSum += dataArray[i]
    trebleSum /= (bufferLength - midEnd)
    for (let i = 0; i < bufferLength; i++) totalSum += dataArray[i]
    
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

  // Audio file loading functions (same as original)
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

    audio.addEventListener('timeupdate', () => setCurrentTime(audio.currentTime))
    audio.addEventListener('ended', () => setIsPlaying(false))

    if (audioElementRef.current) {
      audioElementRef.current.pause()
      if (audioElementRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
    }

    audioElementRef.current = audio

    if (audioContextRef.current) {
      if (sourceRef.current) sourceRef.current.disconnect()
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

    audio.addEventListener('timeupdate', () => setCurrentTime(audio.currentTime))
    audio.addEventListener('ended', () => setIsPlaying(false))

    if (audioElementRef.current) {
      audioElementRef.current.pause()
      if (audioElementRef.current.src.startsWith('blob:')) {
        URL.revokeObjectURL(audioElementRef.current.src)
      }
    }

    audioElementRef.current = audio

    if (audioContextRef.current) {
      if (sourceRef.current) sourceRef.current.disconnect()
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

  // UI event handlers (same as original)
  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('audio/')) loadAudioFile(file)
  }, [loadAudioFile])

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragOver(false)
    const file = event.dataTransfer.files[0]
    if (file && file.type.startsWith('audio/')) loadAudioFile(file)
  }, [loadAudioFile])

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback(() => setIsDragOver(false), [])

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
    if (audioElementRef.current) audioElementRef.current.volume = newVolume
  }, [])

  const formatTime = useCallback((time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }, [])

  const defaultSongs = [
    { filename: "Charli XCX - party 4 u.mp3", displayName: "Charli XCX - Party 4 U" },
    { filename: "Benson Boone - Beautiful Things.mp3", displayName: "Benson Boone - Beautiful Things" },
    { filename: "M83 - Midnight City.mp3", displayName: "M83 - Midnight City" }
  ]

  const loadDefaultSong = useCallback((filename: string, displayName: string) => {
    const url = `/songs/${encodeURIComponent(filename)}`
    loadAudioFromUrl(url, displayName)
  }, [loadAudioFromUrl])

  // WebGL setup with enhanced crystal shader
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

    // Enhanced crystal fragment shader with sharper, more vivid visuals
    const fragmentShaderSource = `#version 300 es
  precision highp float;
  out vec4 outColor;
  
  uniform vec2 u_resolution;
  uniform float u_time;
  uniform float u_audioLevel;
  uniform float u_bassLevel;
  uniform float u_midLevel;
  uniform float u_trebleLevel;
  uniform float u_frequencyData[64];
  uniform float u_waveformData[32];
  uniform float u_intensity;
  uniform float u_rotationSpeed;
  uniform float u_colorSensitivity;
  uniform float u_beatPulse;
  uniform float u_fractalComplexity;
  uniform float u_scaleReactivity;
  
  // Enhanced HSV to RGB with higher saturation
  vec3 hsv(float h, float s, float v) {
    vec4 t = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + t.xyz) * 6.0 - vec3(t.w));
    return v * mix(vec3(t.x), clamp(p - vec3(t.x), 0.0, 1.0), s);
  }
  
  float getFrequency(float pos) {
    int index = int(pos * 63.0);
    return u_frequencyData[index];
  }
  
  void main() {
    vec2 r = u_resolution;
    vec2 FC = gl_FragCoord.xy;
    float t = u_time;
    vec4 o = vec4(0, 0, 0, 1);
    
    // Enhanced audio-reactive scaling with sharper response
    float audioScale = 1.0 + (pow(u_audioLevel, 1.5) * 0.8 + pow(u_bassLevel, 2.0) * 0.6) * u_scaleReactivity * u_intensity;
    
    // Higher iteration count for sharper details
    float maxIterations = 45.0 + u_audioLevel * 15.0 * u_fractalComplexity;
    
    for(float i=0.,g=0.,e=0.,s=0.; i < maxIterations; i++){
      // Higher precision 3D transformation
      vec3 p = vec3((FC.xy - 0.5 * r) / r.y * 1.2 * audioScale, g - 0.3);
      
      // Multi-axis audio-reactive rotation with sharper response
      float bassRot = t * (0.6 + pow(u_bassLevel, 1.5) * 0.5 * u_intensity) * u_rotationSpeed;
      float midRot = t * (0.3 + pow(u_midLevel, 2.0) * 0.4 * u_intensity) * u_rotationSpeed;
      float trebleRot = t * pow(u_trebleLevel, 1.8) * 0.6 * u_rotationSpeed * u_intensity;
      
      // Sharp rotation matrices
      p.yz *= mat2(cos(bassRot), sin(bassRot), -sin(bassRot), cos(bassRot));
      p.xz *= mat2(cos(midRot + trebleRot), sin(midRot + trebleRot), -sin(midRot + trebleRot), cos(midRot + trebleRot));
      p.xy *= mat2(cos(trebleRot * 0.7), sin(trebleRot * 0.7), -sin(trebleRot * 0.7), cos(trebleRot * 0.7));
      
      s = 1.0;
      
      // Enhanced folding parameters for sharper crystals
      vec3 foldParams = vec3(
        2.8 + pow(u_bassLevel, 1.2) * 0.8 * u_intensity, 
        3.5 + pow(u_midLevel, 1.5) * 1.2 * u_intensity, 
        0.8 + pow(u_trebleLevel, 1.3) * 0.6 * u_intensity
      );
      
      vec3 foldOffset = vec3(
        3.2 + pow(u_audioLevel, 1.4) * 0.6 * u_intensity, 
        1.5 + getFrequency(0.15) * 0.8 * u_intensity, 
        1.8 + getFrequency(0.85) * 0.7 * u_intensity
      );
      
      // Enhanced folding with more iterations for sharper edges
      for(int j = 0; j++ < 16; p = foldParams - abs(abs(p) * e - foldOffset))
        s *= e = max(1.005, 9.5 / dot(p, p));
      
      // Sharper geometric accumulation
      float geom = mod(length(p.zx), p.y) / s;
      g += geom * (1.0 + pow(u_audioLevel, 1.3) * 0.25 * u_intensity);
      
      s = log(s) / g;
      
      // Enhanced frequency-based coloring
      float freqPos = i / maxIterations;
      float freqIntensity = getFrequency(freqPos);
      
      // Vivid multi-layered color mapping with higher contrast
      float hue1 = 0.65 - p.z * 0.15 + pow(u_bassLevel, 1.2) * 0.2 * u_colorSensitivity;
      float hue2 = 0.85 + pow(u_midLevel, 1.4) * 0.25 * u_colorSensitivity - freqIntensity * 0.15 * u_colorSensitivity;
      float hue3 = 0.15 + pow(u_trebleLevel, 1.6) * 0.35 * u_colorSensitivity;
      
      // Dynamic hue blending with sharper transitions
      float hue = mix(
        mix(hue1, hue2, pow(u_midLevel * 0.8, 1.2) * u_colorSensitivity), 
        hue3, 
        pow(u_trebleLevel * 0.6, 1.3) * u_colorSensitivity
      );
      
      // Enhanced saturation and brightness for vivid colors
      float saturation = 0.95 + u_audioLevel * 0.05 * u_colorSensitivity;
      float brightness = s / (2800.0 - u_audioLevel * 1200.0 * u_intensity) * 
                        (1.0 + freqIntensity * 0.5 * u_colorSensitivity);
      
      // Sharp beat-reactive brightness pulses
      float beatPulseEffect = 1.0 + pow(u_bassLevel, 2.5) * 0.8 * u_beatPulse;
      brightness *= beatPulseEffect;
      
      // Add sharp frequency spikes
      brightness += pow(freqIntensity, 2.0) * 0.3 * u_intensity;
      
      o.rgb += hsv(hue, saturation, brightness);
    }
    
    // Final enhancement for vivid output
    o.rgb *= 1.0 + pow(u_audioLevel, 1.2) * 0.3 * u_intensity;
    
    // Sharp contrast enhancement
    o.rgb = pow(o.rgb, vec3(0.85));
    
    // Add crystalline sparkles based on high frequencies
    vec2 sparklePos = FC.xy / r;
    float sparkle = 0.0;
    for(int k = 0; k < 8; k++) {
      float freq = getFrequency(float(k) / 7.0);
      vec2 pos = vec2(sin(t * (1.0 + float(k)) + freq * 10.0), cos(t * (1.2 + float(k)) + freq * 8.0)) * 0.3 + 0.5;
      float dist = distance(sparklePos, pos);
      if(dist < 0.01 + freq * 0.02) {
        sparkle += (1.0 - dist / (0.01 + freq * 0.02)) * freq * 2.0;
      }
    }
    
    o.rgb += vec3(sparkle * 0.8, sparkle * 0.9, sparkle) * u_intensity;
    
    outColor = o;
  }
`

    // Shader compilation and setup (same as original)
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
    rotationSpeedUniformLocationRef.current = gl.getUniformLocation(program, "u_rotationSpeed")
    colorSensitivityUniformLocationRef.current = gl.getUniformLocation(program, "u_colorSensitivity")
    beatPulseUniformLocationRef.current = gl.getUniformLocation(program, "u_beatPulse")
    fractalComplexityUniformLocationRef.current = gl.getUniformLocation(program, "u_fractalComplexity")
    scaleReactivityUniformLocationRef.current = gl.getUniformLocation(program, "u_scaleReactivity")

    const startTime = performance.now()

    const render = () => {
      if (!programRef.current || !gl) return
      gl.useProgram(programRef.current)

      const audioData = analyzeAudio()
      const currentTime = (performance.now() - startTime) / 1000

      if (timeUniformLocationRef.current) gl.uniform1f(timeUniformLocationRef.current, currentTime)
      if (resolutionUniformLocationRef.current) gl.uniform2f(resolutionUniformLocationRef.current, canvas.width, canvas.height)
      if (audioLevelUniformLocationRef.current) gl.uniform1f(audioLevelUniformLocationRef.current, audioData.level)
      if (bassLevelUniformLocationRef.current) gl.uniform1f(bassLevelUniformLocationRef.current, audioData.bassLevel)
      if (midLevelUniformLocationRef.current) gl.uniform1f(midLevelUniformLocationRef.current, audioData.midLevel)
      if (trebleLevelUniformLocationRef.current) gl.uniform1f(trebleLevelUniformLocationRef.current, audioData.trebleLevel)
      if (frequencyDataUniformLocationRef.current) gl.uniform1fv(frequencyDataUniformLocationRef.current, audioData.frequencyData)
      if (waveformDataUniformLocationRef.current) gl.uniform1fv(waveformDataUniformLocationRef.current, audioData.waveformData)
      if (intensityUniformLocationRef.current) gl.uniform1f(intensityUniformLocationRef.current, intensity)
      if (rotationSpeedUniformLocationRef.current) gl.uniform1f(rotationSpeedUniformLocationRef.current, rotationSpeed)
      if (colorSensitivityUniformLocationRef.current) gl.uniform1f(colorSensitivityUniformLocationRef.current, colorSensitivity)
      if (beatPulseUniformLocationRef.current) gl.uniform1f(beatPulseUniformLocationRef.current, beatPulse)
      if (fractalComplexityUniformLocationRef.current) gl.uniform1f(fractalComplexityUniformLocationRef.current, fractalComplexity)
      if (scaleReactivityUniformLocationRef.current) gl.uniform1f(scaleReactivityUniformLocationRef.current, scaleReactivity)

      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.drawArrays(gl.TRIANGLES, 0, 6)

      animationRef.current = requestAnimationFrame(render)
    }

    render()

    return () => {
      window.removeEventListener("resize", resizeCanvas)
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      if (program) gl.deleteProgram(program)
      gl.deleteShader(vertexShader)
      gl.deleteShader(fragmentShader)
      gl.deleteBuffer(positionBuffer)
    }
  }, [analyzeAudio, intensity, rotationSpeed, colorSensitivity, beatPulse, fractalComplexity, scaleReactivity])

  useEffect(() => {
    return () => {
      if (audioElementRef.current) {
        audioElementRef.current.pause()
        if (audioElementRef.current.src.startsWith('blob:')) {
          URL.revokeObjectURL(audioElementRef.current.src)
        }
      }
      if (audioContextRef.current) audioContextRef.current.close()
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

      {/* Same UI controls as original with updated title */}
      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          background: "rgba(0, 0, 0, 0.8)",
          borderRadius: "10px",
          padding: "15px",
          color: "white",
          fontFamily: "sans-serif",
          fontSize: "14px",
          minWidth: "300px",
          zIndex: 10,
          backdropFilter: "blur(10px)",
          border: "1px solid rgba(255, 255, 255, 0.1)"
        }}
      >
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
              background: "rgba(255, 255, 255, 0.1)",
              border: "1px solid rgba(255, 255, 255, 0.3)",
              borderRadius: "5px",
              color: "white",
              fontSize: "12px"
            }}
          />
        </div>

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
                  background: "rgba(255, 255, 255, 0.1)",
                  border: "1px solid rgba(255, 255, 255, 0.3)",
                  borderRadius: "5px",
                  color: "white",
                  fontSize: "12px",
                  textAlign: "left",
                  cursor: "pointer",
                  transition: "background-color 0.2s"
                }}
                onMouseEnter={(e) => {
                  (e.target as HTMLButtonElement).style.background = "rgba(255, 255, 255, 0.2)"
                }}
                onMouseLeave={(e) => {
                  (e.target as HTMLButtonElement).style.background = "rgba(255, 255, 255, 0.1)"
                }}
              >
                {song.displayName}
              </button>
            ))}
          </div>
        </div>

        {trackName && (
          <div style={{ marginBottom: "15px", fontSize: "13px", fontWeight: "bold" }}>
            {trackName}
          </div>
        )}

        <div style={{ marginBottom: "15px", textAlign: "center" }}>
          <button
            onClick={togglePlayPause}
            disabled={!audioElementRef.current}
            style={{
              padding: "10px 20px",
              background: isPlaying ? "#ff4444" : "#44ff44",
              border: "none",
              borderRadius: "20px",
              color: "white",
              fontWeight: "bold",
              cursor: "pointer",
              fontSize: "14px",
              opacity: !audioElementRef.current ? 0.5 : 1
            }}
          >
            {isPlaying ? "⏸ Pause" : "▶ Play"}
          </button>
        </div>

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
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>
        )}

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
              background: "rgba(255, 255, 255, 0.3)",
              outline: "none",
              borderRadius: "2px"
            }}
          />
        </div>

        <div style={{ marginBottom: "15px", borderTop: "1px solid rgba(255, 255, 255, 0.2)", paddingTop: "15px" }}>
          <div style={{ marginBottom: "8px", fontSize: "13px", fontWeight: "bold", opacity: 0.9 }}>
            Crystal Controls
          </div>

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
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Rotation Speed: {Math.round(rotationSpeed * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={rotationSpeed * 100}
              onChange={(e) => setRotationSpeed(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Color Vividness: {Math.round(colorSensitivity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={colorSensitivity * 100}
              onChange={(e) => setColorSensitivity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Beat Pulse: {Math.round(beatPulse * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={beatPulse * 100}
              onChange={(e) => setBeatPulse(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Crystal Complexity: {Math.round(fractalComplexity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="150"
              value={fractalComplexity * 100}
              onChange={(e) => setFractalComplexity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "3px", fontSize: "11px", opacity: 0.8 }}>
              Scale Reactivity: {Math.round(scaleReactivity * 100)}%
            </label>
            <input
              type="range"
              min="0"
              max="200"
              value={scaleReactivity * 100}
              onChange={(e) => setScaleReactivity(parseFloat(e.target.value) / 100)}
              style={{
                width: "100%",
                height: "3px",
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

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
                background: "rgba(255, 255, 255, 0.3)",
                outline: "none",
                borderRadius: "2px"
              }}
            />
          </div>

          <div style={{ textAlign: "center", marginTop: "10px" }}>
            <button
              onClick={() => {
                setIntensity(0.5)
                setRotationSpeed(0.5)
                setColorSensitivity(0.5)
                setBeatPulse(0.5)
                setFractalComplexity(0.5)
                setSmoothing(0.85)
                setScaleReactivity(0.5)
              }}
              style={{
                padding: "6px 12px",
                background: "rgba(255, 255, 255, 0.2)",
                border: "1px solid rgba(255, 255, 255, 0.3)",
                borderRadius: "4px",
                color: "white",
                fontSize: "11px",
                cursor: "pointer"
              }}
            >
              Reset Controls
            </button>
          </div>
        </div>

        {isPlaying && (
          <div style={{ fontSize: "10px", opacity: 0.7 }}>
            <div>Level: {Math.round(smoothedAudioDataRef.current.level * 100)}%</div>
            <div>Bass: {Math.round(smoothedAudioDataRef.current.bassLevel * 100)}%</div>
            <div>Mid: {Math.round(smoothedAudioDataRef.current.midLevel * 100)}%</div>
            <div>Treble: {Math.round(smoothedAudioDataRef.current.trebleLevel * 100)}%</div>
          </div>
        )}
      </div>

      {isDragOver && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            background: "rgba(0, 100, 255, 0.2)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 20,
            fontSize: "24px",
            color: "white",
            fontFamily: "sans-serif",
            fontWeight: "bold"
          }}
        >
          Drop audio file here
        </div>
      )}
    </div>
  )
} 