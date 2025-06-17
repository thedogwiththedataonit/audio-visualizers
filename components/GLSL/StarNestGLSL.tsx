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
 * StarNest GLSL Visualization Component
 *
 * A React component that renders an audio-reactive star nebula visualization
 * with volumetric rendering and customizable visual controls.
 */
const StarNestGLSL: React.FC = () => {
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
  
  // Visual control uniform locations
  const audioIntensityUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const zoomUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const brightnessUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const darkmatterUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const saturationUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const formuparamUniformLocationRef = useRef<WebGLUniformLocation | null>(null)

  // Audio state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.7)
  const [trackName, setTrackName] = useState("")
  const [isDragOver, setIsDragOver] = useState(false)

  // Visual controls
  const [audioIntensity, setAudioIntensity] = useState(0.5)
  const [zoom, setZoom] = useState(0.8)
  const [brightness, setBrightness] = useState(0.0015)
  const [darkmatter, setDarkmatter] = useState(0.3)
  const [saturation, setSaturation] = useState(0.85)
  const [formuparam, setFormuparam] = useState(0.53)

  // Audio data
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
    analyser.getByteFrequencyData(dataArray)

    // Calculate frequency bands
    const bassEnd = Math.floor(bufferLength * 0.1)
    const midEnd = Math.floor(bufferLength * 0.5)

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

    // Smooth the audio data
    const smoothingFactor = 0.85
    const current = smoothedAudioDataRef.current

    current.level = current.level * smoothingFactor + level * (1 - smoothingFactor)
    current.bassLevel = current.bassLevel * smoothingFactor + (bassSum / 255) * (1 - smoothingFactor)
    current.midLevel = current.midLevel * smoothingFactor + (midSum / 255) * (1 - smoothingFactor)
    current.trebleLevel = current.trebleLevel * smoothingFactor + (trebleSum / 255) * (1 - smoothingFactor)

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

      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.3

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

      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.3

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

// Visual control uniforms
uniform float u_audioIntensity;
uniform float u_zoom;
uniform float u_brightness;
uniform float u_darkmatter;
uniform float u_saturation;
uniform float u_formuparam;

#define iterations 7
#define volsteps 20
#define stepsize 0.1
#define tile 0.850
#define speed 0.000
#define distfading 0.730

float happy_star(vec2 uv, float anim)
{
    uv = abs(uv);
    vec2 pos = min(uv.xy/uv.yx, anim);
    float p = (2.0 - pos.x - pos.y);
    return (2.0+p*(p*p-1.5)) / (uv.x+uv.y);      
}

void main() {
    // Get coords and direction
    vec2 uv = gl_FragCoord.xy/iResolution.xy - 0.5;
    uv = normalize(vec3(uv, 0.31)).xy;
    uv.y *= iResolution.y/iResolution.x;
    
    // Audio-reactive zoom
    float audioZoom = u_zoom + u_bassLevel * 0.3 * u_audioIntensity;
    vec3 dir = normalize(vec3(uv * audioZoom, 0.45));
    
    float time = iTime * speed + 0.25;
    dir.z += iTime * 0.1 + u_midLevel * 0.2 * u_audioIntensity;
    
    // Rotation with audio
    float rotSpeed = 0.1 + u_midLevel * 0.3 * u_audioIntensity;
    float t = iTime * rotSpeed + ((.25 + .05 * sin(iTime * .1))/(length(uv.xy) + .57)) * 1.2;
    float si = sin(t);
    float co = cos(t);
    mat2 ma = mat2(co, si, -si, co);
    
    uv *= ma;
    
    float time2 = (-iTime) * 10.0;
    float s2 = 0.0, v = 0.0;
    vec2 uv2 = (-iResolution.xy + 2.0 * gl_FragCoord.xy) / iResolution.y;
    float t2 = time * 0.005;
    
    vec3 col2 = vec3(0.0);
    vec3 init = vec3(0.25, 0.25 + sin(time * 0.001) * .1, time * 0.0008);
    
    for (int r = 0; r < 100; r++) {
        vec3 p = init + s2 * vec3(uv, 0.143);
        p.xy *= ma;
        p.z = mod(p.z, 2.0);
        for (int i = 0; i < 10; i++) p = abs(p * 2.04) / dot(p, p) - 0.75;
        v += length(p * p) * smoothstep(0.0, 0.5, 0.9 - s2) * .002;
        col2 += vec3(v * 0.8, 1.1 - s2 * 0.5, .7 + v * 0.5) * v * 0.013;
        s2 += .01;
    }
    
    float v1, v2, v3;
    v1 = v2 = v3 = 0.0;
    
    float s = 0.0;
    for (int i = 0; i < 90; i++) {
        vec3 p = s * normalize(vec3(uv, 0.1)) + col2;
        p += vec3(.22, .3, s - 1.5 - sin(iTime * .13) * .1);
        for (int i = 0; i < 8; i++) p = abs(p) / dot(p,p) - 0.659;
        v1 += dot(p,p) * .0015 * (1.8 + sin(length(uv.xy * 13.0) + .5  - iTime * .2));
        v2 += dot(p,p) * .0013 * (1.5 + sin(length(uv.xy * 14.5) + 1.2 - iTime * .3));
        v3 += length(p.xy*10.) * .0003;
        s += .035;
    }
    
    float len = length(uv);
    v1 *= smoothstep(.7, .0, len);
    v2 *= smoothstep(.5, .0, len);
    v3 *= smoothstep(.9, .0, len);
    
    vec3 col = vec3(v3 * (1.5 + sin(iTime * .2) * .4),
                    (v1 + v3) * .3,
                    v2) + smoothstep(0.2, .0, len) * .35 + smoothstep(.0, .6, v3) * .15;
    
    vec4 fc2 = vec4(min(pow(abs(col)*col2*22., vec3(1.2)), 1.0), 1.0);
    
    // Volumetric rendering
    vec3 from = vec3(1., .5, 0.5);
    float s5 = 0.1, fade = 1.;
    vec3 v5 = vec3(0.);
    
    // Audio-reactive formuparam
    float audioFormuparam = u_formuparam + u_trebleLevel * 0.1 * u_audioIntensity;
    
    for (int r = 0; r < volsteps; r++) {
        vec3 p = from + s5 * dir * .5;
        p = abs(vec3(tile) - mod(p, vec3(tile * 2.)));
        float pa, a = pa = 0.;
        
        for (int i = 0; i < iterations; i++) {
            p = abs(p) / dot(p, p) - audioFormuparam;
            // Audio-reactive rotation
            float rotAngle = iTime * 0.04 + u_midLevel * 0.1 * u_audioIntensity;
            p.xy *= mat2(cos(rotAngle), sin(rotAngle), -sin(rotAngle), cos(rotAngle));
            a += abs(length(p) - pa);
            pa = length(p);
        }
        
        // Audio-reactive dark matter
        float audioDarkmatter = u_darkmatter - u_bassLevel * 0.1 * u_audioIntensity;
        float dm = max(0., audioDarkmatter - a * a * .001);
        a *= a * a;
        if (r > 6) fade *= 1.3 - dm;
        v5 += fade;
        
        // Audio-reactive brightness
        float audioBrightness = u_brightness * (1.0 + u_audioLevel * 2.0 * u_audioIntensity);
        v5 += vec3(s5, s5*s5, s5*s5*s5*s5) * a * audioBrightness * fade;
        fade *= distfading;
        s5 += stepsize;
    }
    
    // Audio-reactive saturation
    float audioSaturation = u_saturation + u_trebleLevel * 0.2 * u_audioIntensity;
    v5 = mix(vec3(length(v5)), v5, audioSaturation);
    
    fragColor = vec4(v5 * .03 * fc2.xyz * 2., 1.);
    
    // Star effect with audio - reduced intensity
    uv *= 2.0 * (cos(iTime * 2.0) - 2.5);
    float anim = sin(iTime * 12.0 + u_trebleLevel * 20.0 * u_audioIntensity) * 0.1 + 1.0;
    fragColor += vec4(happy_star(uv, anim) * vec3(0.2, 0.2, 0.22) + fc2.xyz * 0.2, 1.0);
    
    // Overall audio brightness boost
    fragColor.rgb *= 1.0 + u_audioLevel * 0.5 * u_audioIntensity;
    
    // Apply vignette to darken center
    float vignette = smoothstep(0.0, 0.7, len);
    fragColor.rgb *= 0.5 + 0.5 * vignette;
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
      
      const audioIntensityUniformLocation = gl.getUniformLocation(program, "u_audioIntensity")
      const zoomUniformLocation = gl.getUniformLocation(program, "u_zoom")
      const brightnessUniformLocation = gl.getUniformLocation(program, "u_brightness")
      const darkmatterUniformLocation = gl.getUniformLocation(program, "u_darkmatter")
      const saturationUniformLocation = gl.getUniformLocation(program, "u_saturation")
      const formuparamUniformLocation = gl.getUniformLocation(program, "u_formuparam")

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

        if (audioIntensityUniformLocation) {
          gl.uniform1f(audioIntensityUniformLocation, audioIntensity)
        }
        if (zoomUniformLocation) {
          gl.uniform1f(zoomUniformLocation, zoom)
        }
        if (brightnessUniformLocation) {
          gl.uniform1f(brightnessUniformLocation, brightness)
        }
        if (darkmatterUniformLocation) {
          gl.uniform1f(darkmatterUniformLocation, darkmatter)
        }
        if (saturationUniformLocation) {
          gl.uniform1f(saturationUniformLocation, saturation)
        }
        if (formuparamUniformLocation) {
          gl.uniform1f(formuparamUniformLocation, formuparam)
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
  }, [analyzeAudio, audioIntensity, zoom, brightness, darkmatter, saturation, formuparam])

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
          StarNest Audio Visualizer
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

          {/* Zoom */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Zoom: {zoom.toFixed(2)}
            </label>
            <input
              type="range"
              min="20"
              max="150"
              value={zoom * 100}
              onChange={(e) => setZoom(parseFloat(e.target.value) / 100)}
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

          {/* Brightness */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Brightness: {(brightness * 1000).toFixed(1)}
            </label>
            <input
              type="range"
              min="5"
              max="30"
              value={brightness * 10000}
              onChange={(e) => setBrightness(parseFloat(e.target.value) / 10000)}
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

          {/* Dark Matter */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Dark Matter: {darkmatter.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={darkmatter * 100}
              onChange={(e) => setDarkmatter(parseFloat(e.target.value) / 100)}
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

          {/* Saturation */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Saturation: {saturation.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={saturation * 100}
              onChange={(e) => setSaturation(parseFloat(e.target.value) / 100)}
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

          {/* Form Parameter */}
          <div style={{ marginBottom: "12px" }}>
            <label style={{ display: "block", marginBottom: "4px", fontSize: "11px", opacity: 0.7 }}>
              Form Complexity: {formuparam.toFixed(2)}
            </label>
            <input
              type="range"
              min="30"
              max="80"
              value={formuparam * 100}
              onChange={(e) => setFormuparam(parseFloat(e.target.value) / 100)}
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
                setAudioIntensity(0.5)
                setZoom(0.8)
                setBrightness(0.0015)
                setDarkmatter(0.3)
                setSaturation(0.85)
                setFormuparam(0.53)
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

export default StarNestGLSL 