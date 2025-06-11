"use client"

import { useRef, useEffect } from "react"

export default function MountainsGLSLVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const programRef = useRef<WebGLProgram | null>(null)
  const resolutionUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const mouseUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const timeUniformLocationRef = useRef<WebGLUniformLocation | null>(null)
  const mouseXRef = useRef(0)
  const mouseYRef = useRef(0)
  const elapsedTimeRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Get WebGL2 context
    const gl = canvas.getContext("webgl2")
    if (!gl) {
      console.error("WebGL2 not supported")
      return
    }

    // Handle canvas resize
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
      gl.viewport(0, 0, canvas.width, canvas.height)
    }

    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Vertex shader source
    const vertexShaderSource = `#version 300 es
    in vec4 a_position;
    void main() {
      gl_Position = a_position;
    }
  `

    // Fragment shader source - with detailed explanations
    const fragmentShaderSource = `#version 300 es
precision highp float;
out vec4 outColor;
uniform vec2 u_resolution;  // Canvas resolution (width, height)
uniform vec2 u_mouse;       // Mouse position
uniform float u_time;       // Time in seconds

// HSV to RGB color conversion function
// h: hue [0,1], s: saturation [0,1], v: value/brightness [0,1]
// Returns RGB color in range [0,1]
vec3 hsv(float h,float s,float v){
  vec4 t=vec4(1.,2./3.,1./3.,3.);
  vec3 p=abs(fract(vec3(h)+t.xyz)*6.-vec3(t.w));
  return v*mix(vec3(t.x),clamp(p-vec3(t.x),0.,1.),s);
}

void main(){
  // Initialize variables
  vec2 r=u_resolution;                  // Screen resolution
  vec2 FC=gl_FragCoord.xy;              // Current fragment coordinates
  float t=u_time;                       // Time variable for animation
  vec4 o=vec4(0,0,0,1);                 // Output color (initially black with alpha=1)
  
  // Declare variables for the fractal generation
  float i,e,g,R,s;
  
  // Ray direction vector - normalized to create perspective effect
  // Maps screen coordinates to [-0.5,0.5] range and adds z component
  vec3 q,p,d=vec3((FC.xy-.5*r)/r,.52);  // d is the initial ray direction
  
  // Main ray marching loop - iterates 99 times to build the image
  for(q.zy--;i++<99.;){
    // Accumulate a tiny amount based on iteration count - creates depth effect
    e+=i/9e9;
    
    // Accumulate color based on current position and iteration
    // This creates the colorful glow effect using HSV color space
    // p.y controls hue, q.y controls saturation, and e*i (limited) controls brightness
    o.rgb+=hsv(p.y,q.y,min(e*i,.01));
    
    // Reset scale factor for the fractal iteration
    s=2.;
    
    // Update position - ray marching step
    // R is the length of the current position vector
    // This creates the movement through the fractal space
    p=q+=d*e*R*.25;
    
    // Accumulate height value - affects overall brightness distribution
    g+=p.y/s;
    
    // Transform position vector using logarithmic and exponential functions
    // This creates the fractal-like structures and time-based animation
    // log2(R) creates concentric circular patterns
    // exp2(mod(-p.z,s)/R) creates radial patterns
    p=vec3(log2(R=length(p))-t*.2,exp2(mod(-p.z,s)/R),p);
    
    // Inner loop - fractal detail generation using frequency doubling (s+=s)
    // This is a form of fractal Brownian motion (fBm) or multi-octave noise
    // Creates detailed noise patterns by summing sine/cosine waves at different frequencies
    for(e=--p.y;s<6e3;s+=s)
      // The dot product between sine and cosine creates interference patterns
      // This is similar to Perlin noise but more compact
      // Dividing by frequency (s) creates a 1/f noise spectrum (pink noise)
      // The result is a complex noise field with fractal characteristics
      e+=-abs(dot(sin(p.xzz*s),cos(p.zzy*s))/s*.6);
  }
  
  // Final output color
  outColor=o;
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

    // Create vertex and fragment shaders
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource)
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource)

    if (!vertexShader || !fragmentShader) return

    // Create and link program
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

    // Set up position buffer (a rectangle covering the entire canvas)
    const positionBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)

    const positions = [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW)

    // Set up attributes
    const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
    gl.enableVertexAttribArray(positionAttributeLocation)
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0)

    // Get uniform locations
    const resolutionUniformLocation = gl.getUniformLocation(program, "u_resolution")
    const mouseUniformLocation = gl.getUniformLocation(program, "u_mouse")
    const timeUniformLocation = gl.getUniformLocation(program, "u_time")

    // Mouse tracking
    let mouseX = 0
    let mouseY = 0

    const handleMouseMove = (event: MouseEvent) => {
      mouseX = event.clientX
      mouseY = event.clientY
    }

    window.addEventListener("mousemove", handleMouseMove)

    // Render loop
    const startTime = performance.now()
    let animationFrameId: number

    // Use the program outside the render loop
    gl.useProgram(program)

    const render = () => {
      const currentTime = performance.now()
      const elapsedTime = (currentTime - startTime) / 1000 // Convert to seconds

      // Update uniforms
      if (resolutionUniformLocation) {
        gl.uniform2f(resolutionUniformLocation, canvas.width, canvas.height)
      }
      if (mouseUniformLocation) {
        gl.uniform2f(mouseUniformLocation, mouseX, mouseY)
      }
      if (timeUniformLocation) {
        gl.uniform1f(timeUniformLocation, elapsedTime)
      }

      // Clear and draw
      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.drawArrays(gl.TRIANGLES, 0, 6)

      // Continue animation loop
      animationFrameId = requestAnimationFrame(render)
    }

    // Start the render loop
    render()

    // Cleanup
    return () => {
      cancelAnimationFrame(animationFrameId)
      window.removeEventListener("resize", resizeCanvas)
      window.removeEventListener("mousemove", handleMouseMove)

      // Clean up WebGL resources
      gl.deleteProgram(program)
      gl.deleteShader(vertexShader)
      gl.deleteShader(fragmentShader)
      gl.deleteBuffer(positionBuffer)
    }
  }, [])

  return (
    <div className="relative w-screen h-screen">
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
      <div
        className="absolute bottom-4 right-4 text-white text-sm font-light z-10"
        style={{ textShadow: "0px 0px 3px rgba(0,0,0,0.5)" }}
      >
        <a
          href="https://x.com/YoheiNishitsuji/status/1920826093746958747"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:underline"
        >
          @Yohei Nishitsuji
        </a>
      </div>
    </div>
  )
}
