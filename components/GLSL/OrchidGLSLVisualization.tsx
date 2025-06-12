import React, { useRef, useEffect } from 'react';

const OrchidGLSLVisualization: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl2');
    if (!gl) {
      console.error('WebGL2 not supported');
      return;
    }

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      gl.viewport(0, 0, canvas.width, canvas.height);
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    const vertexShaderSource = `#version 300 es
      in vec4 a_position;
      void main() {
        gl_Position = a_position;
      }
    `;

    const fragmentShaderSource = `#version 300 es
      precision highp float;
      out vec4 outColor;
      uniform vec2 u_resolution;
      uniform vec2 u_mouse;
      uniform float u_time;

      // HSV to RGB conversion function
      // h: hue, s: saturation, v: value
      vec3 hsv(float h,float s,float v){
        vec4 t=vec4(1.,2./3.,1./3.,3.);
        vec3 p=abs(fract(vec3(h)+t.xyz)*6.-vec3(t.w));
        return v*mix(vec3(t.x),clamp(p-vec3(t.x),0.,1.),s);
      }

      // 2D rotation matrix
      mat2 rotate2D(float angle) {
        return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
      }

      void main(){
        vec2 r=u_resolution;
        vec2 FC=gl_FragCoord.xy;
        float t=u_time;
        vec4 o=vec4(0,0,0,1);

        // Main loop for generating the fractal-like pattern
        // This loop implements a form of ray marching combined with domain warping
        for(float i=0.,g=0.,e=0.,s=0.;++i<85.;o.rgb+=hsv(g*i*.1-.5,e,s/5e2)){
          // Initialize ray position
          // The vec2(0,1.1) offset creates a vertical shift in the pattern
          vec3 p=vec3((FC.xy-.5*r)/r.y+vec2(0,1.1),g+.1);
          
          // Rotate the xy plane over time
          p.zx*=rotate2D(t*.5);
          
          s=2.;
          
          // Inner loop for fractal iteration
          // This creates a Mandelbox-like fractal structure
          for(int i=0;i++<12;p=vec3(2,5,2)-abs(abs(p)*e-vec3(5,4,4)))
            // The dot product here acts as a distance estimator
            // It creates a noise-like pattern due to the repeated folding and scaling
            s*=e=max(1.02,12./dot(p,p));
          
          // Accumulate color based on the fractal iteration
          // The mod operation creates a repeating pattern in the color space
          g+=mod(length(p.xz),p.y)/s;
          
          // Adjust the scale factor for the next iteration
          // This logarithmic scaling creates a sense of depth in the visualization
          s=log2(s*.2);
        }
        
        outColor=o;
      }

      // Notes on the visualization:
      // - This shader implements a form of ray marching through a fractal-like structure
      // - The inner loop creates a Mandelbox-inspired fractal pattern
      // - The noise-like effect comes from the repeated folding and scaling operations
      // - It's not a traditional noise function (like Perlin or Simplex), but rather
      //   emergent complexity from the fractal iterations
      // - The color is determined by the accumulated 'g' value, creating a psychedelic effect
      // - The time-based rotation adds dynamic movement to the visualization
    `;

    const createShader = (gl: WebGL2RenderingContext, type: number, source: string) => {
      const shader = gl.createShader(type);
      if (!shader) {
        console.error('An error occurred creating the shaders');
        return null;
      }
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('An error occurred compiling the shaders: ' + gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

    if (!vertexShader || !fragmentShader) return;

    const program = gl.createProgram();
    if (!program) {
      console.error('An error occurred creating the shader program');
      return;
    }

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error('Unable to initialize the shader program: ' + gl.getProgramInfoLog(program));
      return;
    }

    gl.useProgram(program);

    const positionAttributeLocation = gl.getAttribLocation(program, 'a_position');
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
      -1,  1,
       1, -1,
       1,  1,
    ]);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

    const resolutionUniformLocation = gl.getUniformLocation(program, 'u_resolution');
    const mouseUniformLocation = gl.getUniformLocation(program, 'u_mouse');
    const timeUniformLocation = gl.getUniformLocation(program, 'u_time');

    let mouseX = 0;
    let mouseY = 0;

    canvas.addEventListener('mousemove', (event) => {
      mouseX = event.clientX;
      mouseY = event.clientY;
    });

    const render = (time: number) => {
      gl.uniform2f(resolutionUniformLocation, canvas.width, canvas.height);
      gl.uniform2f(mouseUniformLocation, mouseX, mouseY);
      gl.uniform1f(timeUniformLocation, time * 0.001);

      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.drawArrays(gl.TRIANGLES, 0, 6);

      requestAnimationFrame(render);
    };

    requestAnimationFrame(render);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      gl.deleteProgram(program);
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
    };
  }, []);

  return (
    <>
      <canvas ref={canvasRef} style={{ width: '100vw', height: '100vh' }} />
    </>
  );
};

export default OrchidGLSLVisualization;

