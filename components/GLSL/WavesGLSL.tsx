"use client"

import { useRef, useEffect } from "react"

/**
 * TweetGLSLVisualization
 *
 * A React component that renders a WebGL visualization based on the つぶやきGLSL (Tweet GLSL) by Yohei Nishitsuji
 * Original tweet: https://x.com/YoheiNishitsuji/status/1898534760387064316
 */
export default function TweetGLSLVisualization() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // Get WebGL2 context
    const gl = canvas.getContext("webgl2")
    if (!gl) {
      console.error("WebGL2 not supported")
      return
    }

    // Resize canvas to full screen
    const resizeCanvas = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resizeCanvas()
    window.addEventListener("resize", resizeCanvas)

    // Vertex shader - simple pass-through
    const vertexShaderSource = `#version 300 es
      in vec4 a_position;
      void main() {
        gl_Position = a_position;
      }
    `

    // Fragment shader - with added rotation and detailed comments
    const fragmentShaderSource = `#version 300 es
  precision highp float;
  out vec4 outColor;
  uniform vec2 u_resolution;
  uniform vec2 u_mouse;
  uniform float u_time;
  uniform float u_rotationX;
  uniform float u_rotationY;

  /**
   * HSV to RGB color conversion
   * 
   * A compact implementation of the HSV (Hue, Saturation, Value) to RGB color space conversion.
   * This is used for creating smooth color gradients based on spatial coordinates.
   * 
   * @param h - Hue value [0,1]
   * @param s - Saturation value [0,1]
   * @param v - Value/Brightness [0,1]
   * @return RGB color vector
   */
  vec3 hsv(float h,float s,float v){
    vec4 t=vec4(1.,2./3.,1./3.,3.);
    vec3 p=abs(fract(vec3(h)+t.xyz)*6.-vec3(t.w));
    return v*mix(vec3(t.x),clamp(p-vec3(t.x),0.,1.),s);
  }

  /**
   * Rotation around X axis
   * 
   * Standard 3D rotation matrix for rotating around the X axis.
   * Used to implement interactive view rotation.
   */
  vec3 rotateX(vec3 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
  }

  /**
   * Rotation around Y axis
   * 
   * Standard 3D rotation matrix for rotating around the Y axis.
   * Used to implement interactive view rotation.
   */
  vec3 rotateY(vec3 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
  }

  void main() {
    // Initialize variables
    vec2 r = u_resolution;                          // Screen resolution
    vec2 FC = gl_FragCoord.xy;                      // Fragment coordinates
    float t = u_time;                               // Time uniform for animation
    vec4 o = vec4(0,0,0,1);                         // Output color (initially black)
    
    // Ray marching variables
    float i,e,R,s;                                  // Loop counters, distance estimation, radius, scale
    
    // Ray direction setup - this is the core of the ray marching technique
    // Creates a ray direction vector based on screen coordinates
    vec3 q,p,d=vec3(FC.xy/r*.6+vec2(-.3,.63),1);    // q: position, p: transformed position, d: ray direction
    
    // Apply rotation to the direction vector for interactive viewing
    d = rotateX(d, u_rotationX);                    // Rotate around X axis based on mouse Y position
    d = rotateY(d, u_rotationY);                    // Rotate around Y axis based on mouse X position
    
    // Primary ray marching loop - iterates 110 times to accumulate color along the ray
    // This is a form of volumetric ray marching where we accumulate color contributions
    // rather than stopping at the first hit
    for(q.zy--;i++<110.;){
      // Accumulate color based on current position and parameters
      // The color is determined by a combination of position (p.x), radius (R),
      // and a complex function of distance estimation (e), scale (s), and position (q.z)
      o.rgb+=hsv(.45-p.x,R,min(e*s-q.z,R)/8.);
      
      // Reset scale for the next iteration
      s=3.;
      
      // Ray marching step - advance the ray by a distance proportional to
      // the current distance estimation (e), radius (R), and direction (d)
      p=q+=d*e*R*.2;
      
      // Transform the position for the next iteration
      // This creates a complex spatial transformation that gives the fractal-like appearance
      // R=length(p) calculates the radius, which is then used in multiple ways:
      // 1. Scaled by 7.6 for the x component
      // 2. Combined with an exponential function of -p.z/R for the y component
      // 3. The arctangent of p.y/p.x gives the angle for the z component
      p=vec3((R=length(p))*7.6,exp(-p.z/R)+R-.05,atan(p.y,p.x)*s);
      
      // Secondary loop - creates a domain-warped noise field
      // This is a form of procedural noise that uses sine and cosine functions
      // with domain warping (using the position vector in different permutations)
      // The loop doubles the frequency (s+=s) each iteration, creating an
      // octave-based noise similar to fractional Brownian motion (fBm)
      // but with a custom accumulation function
      for(e=--p.y;s<5e2;s+=s)
        // The dot product creates interference patterns between the sine and cosine waves
        // This is similar to wave interference in signal processing
        // The division by s creates a frequency-dependent amplitude scaling (1/f noise characteristic)
        e+=dot(sin(p.xzy*s+.2+t),.1+cos(p.zyx*s+t))/s*.3;
    }
    
    // Output the final accumulated color
    outColor = o;
  }
`

    let program: WebGLProgram | null = null
    let vertexShader: WebGLShader | null = null
    let fragmentShader: WebGLShader | null = null
    let positionBuffer: WebGLBuffer | null = null

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

    vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource)
    fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource)

    if (!vertexShader || !fragmentShader) return

    // Create program
    program = gl.createProgram()
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

    // Use our program
    gl.useProgram(program)

    // Set up position buffer (full screen quad)
    positionBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW)

    // Set up attributes
    const positionAttributeLocation = gl.getAttribLocation(program, "a_position")
    gl.enableVertexAttribArray(positionAttributeLocation)
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0)

    // Set up uniforms
    const resolutionUniformLocation = gl.getUniformLocation(program, "u_resolution")
    const mouseUniformLocation = gl.getUniformLocation(program, "u_mouse")
    const timeUniformLocation = gl.getUniformLocation(program, "u_time")
    const rotationXUniformLocation = gl.getUniformLocation(program, "u_rotationX")
    const rotationYUniformLocation = gl.getUniformLocation(program, "u_rotationY")

    // Mouse tracking and rotation
    let mouseX = 0
    let mouseY = 0
    let rotationX = 0
    let rotationY = 0
    let targetRotationX = 0
    let targetRotationY = 0

    const handleMouseMove = (e: MouseEvent) => {
      mouseX = e.clientX
      mouseY = e.clientY

      // Calculate normalized coordinates (-1 to 1)
      const normalizedX = (e.clientX / window.innerWidth) * 2 - 1
      const normalizedY = (e.clientY / window.innerHeight) * 2 - 1

      // Add a dead zone in the center (20% of the screen)
      const deadZone = 0.2

      // Apply dead zone and reduced sensitivity
      targetRotationX = (Math.abs(normalizedY) > deadZone ? normalizedY : 0) * 0.1 * Math.PI
      targetRotationY = (Math.abs(normalizedX) > deadZone ? normalizedX : 0) * 0.1 * Math.PI
    }
    window.addEventListener("mousemove", handleMouseMove)

    // Add touch support
    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) {
        const touch = e.touches[0]

        // Calculate normalized coordinates (-1 to 1)
        const normalizedX = (touch.clientX / window.innerWidth) * 2 - 1
        const normalizedY = (touch.clientY / window.innerHeight) * 2 - 1

        // Add a dead zone in the center (20% of the screen)
        const deadZone = 0.2

        // Apply dead zone and reduced sensitivity
        targetRotationX = (Math.abs(normalizedY) > deadZone ? normalizedY : 0) * 0.1 * Math.PI
        targetRotationY = (Math.abs(normalizedX) > deadZone ? normalizedX : 0) * 0.1 * Math.PI
      }
      e.preventDefault()
    }

    window.addEventListener("touchmove", handleTouchMove, { passive: false })

    // Render loop
    const startTime = performance.now()
    let animationFrameId: number

    const render = () => {
      const currentTime = performance.now()
      const elapsedTime = (currentTime - startTime) / 1000 // Convert to seconds

      // Update canvas size if needed
      if (canvas.width !== window.innerWidth || canvas.height !== window.innerHeight) {
        resizeCanvas()
      }

      // Smooth rotation interpolation with greatly reduced speed
      rotationX += (targetRotationX - rotationX) * 0.01
      rotationY += (targetRotationY - rotationY) * 0.01

      // Set viewport and clear
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)

      // Update uniforms
      gl.uniform2f(resolutionUniformLocation, canvas.width, canvas.height)
      gl.uniform2f(mouseUniformLocation, mouseX, canvas.height - mouseY)
      gl.uniform1f(timeUniformLocation, elapsedTime)
      gl.uniform1f(rotationXUniformLocation, rotationX)
      gl.uniform1f(rotationYUniformLocation, rotationY)

      // Draw
      gl.drawArrays(gl.TRIANGLES, 0, 6)

      // Request next frame
      animationFrameId = requestAnimationFrame(render)
    }

    // Start rendering
    render()

    // Cleanup
    return () => {
      cancelAnimationFrame(animationFrameId)
      window.removeEventListener("resize", resizeCanvas)
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("touchmove", handleTouchMove)
      if (program) gl.deleteProgram(program)
      if (vertexShader) gl.deleteShader(vertexShader)
      if (fragmentShader) gl.deleteShader(fragmentShader)
      if (positionBuffer) gl.deleteBuffer(positionBuffer)
    }
  }, [])

  return (
    <div className="relative w-full h-screen">
      <canvas ref={canvasRef} className="w-full h-screen block" style={{ touchAction: "none" }} />
      <div className="absolute bottom-2 right-2 text-white text-xs opacity-50 bg-black bg-opacity-30 px-2 py-1 rounded">
        <a
          href="https://x.com/YoheiNishitsuji/status/1898534760387064316"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:opacity-100"
        >
          @Yohei Nishitsuji
        </a>
      </div>
    </div>
  )
}

