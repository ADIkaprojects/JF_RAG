import { useEffect, useRef } from "react";

interface WaveformVisualizerProps {
  isActive: boolean;
  analyserNode: AnalyserNode | null;
}

const WaveformVisualizer = ({ isActive, analyserNode }: WaveformVisualizerProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const centerY = height / 2;

    let dataArray: Uint8Array | null = null;
    if (analyserNode) {
      analyserNode.fftSize = 256;
      dataArray = new Uint8Array(analyserNode.frequencyBinCount) as any;
    }

    let phase = 0;

    const drawWave = (
      amplitude: number,
      frequency: number,
      phaseOffset: number,
      color: string,
      lineWidth: number,
      opacity: number
    ) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.globalAlpha = opacity;

      for (let x = 0; x <= width; x++) {
        const normalizedX = x / width;
        // Envelope: fade edges
        const envelope = Math.sin(normalizedX * Math.PI);
        const y =
          centerY +
          Math.sin(normalizedX * Math.PI * 2 * frequency + phase + phaseOffset) *
            amplitude *
            envelope;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      ctx.stroke();
      ctx.globalAlpha = 1;
    };

    const animate = () => {
      ctx.clearRect(0, 0, width, height);

      let avgAmplitude = 0;
      if (isActive && analyserNode && dataArray) {
        analyserNode.getByteFrequencyData(dataArray as unknown as Uint8Array<ArrayBuffer>);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
        avgAmplitude = sum / dataArray.length / 255;
      }

      const baseAmplitude = isActive ? 8 + avgAmplitude * 40 : 3;
      phase += isActive ? 0.04 : 0.015;

      // Layer 1: Outer glow waves (subtle)
      drawWave(baseAmplitude * 0.4, 1.5, 0.5, "hsl(250, 70%, 60%)", 1, 0.15);
      drawWave(baseAmplitude * 0.35, 2, -0.8, "hsl(250, 70%, 60%)", 1, 0.1);

      // Layer 2: Mid waves
      drawWave(baseAmplitude * 0.6, 1.8, 1.2, "hsl(200, 80%, 70%)", 1.2, 0.25);
      drawWave(baseAmplitude * 0.5, 2.2, -1.5, "hsl(200, 80%, 70%)", 1, 0.2);

      // Layer 3: Primary waves (bright)
      drawWave(baseAmplitude * 0.9, 1.2, 0, "hsl(217, 91%, 65%)", 2, 0.6);
      drawWave(baseAmplitude * 0.7, 1.6, 2, "hsl(217, 91%, 55%)", 1.5, 0.4);

      // Layer 4: Core wave (brightest)
      drawWave(baseAmplitude, 1, 0.3, "hsl(210, 100%, 80%)", 2.5, 0.8);

      // Glow effect in center
      if (isActive && avgAmplitude > 0.1) {
        const gradient = ctx.createRadialGradient(
          width / 2, centerY, 0,
          width / 2, centerY, width * 0.3
        );
        gradient.addColorStop(0, `hsla(217, 91%, 60%, ${avgAmplitude * 0.15})`);
        gradient.addColorStop(0.5, `hsla(250, 80%, 65%, ${avgAmplitude * 0.05})`);
        gradient.addColorStop(1, "transparent");
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationRef.current);
    };
  }, [isActive, analyserNode]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-32"
      style={{ width: "100%", height: "128px" }}
    />
  );
};

export default WaveformVisualizer;

