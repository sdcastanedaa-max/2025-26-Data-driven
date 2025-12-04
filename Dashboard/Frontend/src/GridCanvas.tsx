// frontend/src/GridCanvas.tsx
import React, { useEffect, useRef } from "react";
import type { GridNode, NodeType } from "./gridData";
import { NODES, LINES } from "./gridData";
import type { LineFlow } from "./simulation";

interface GridCanvasProps {
  tHours: number;
  flows: Record<string, LineFlow>;
  typeVisibility: Record<NodeType, boolean>;
  onHover?: (info: { node: GridNode | null; x: number; y: number }) => void;
  onClickNode?: (info: {
    node: GridNode | null;
    x: number;
    y: number;
  }) => void;
}

const NODE_COLOR: Record<NodeType, string> = {
  substation: "#ffffff",
  wind: "#00aaff",
  pv: "#ffcc00",
  fossil: "#ff6666",
  nuclear: "#9b59b6",
  bess: "#bb66ff",
  load_res: "#66ff99",
  load_ind: "#ff9966",
};

const ICON_PATHS: Partial<Record<NodeType, string>> = {
  pv: "/icons/pv.png",
  wind: "/icons/wind.png",
  fossil: "/icons/fossil.png",
  nuclear: "/icons/nuclear.png",
  bess: "/icons/bess.png",
  substation: "/icons/substation.png",
  load_res: "/icons/load_res.png",
  load_ind: "/icons/load_ind.png",
};

const GridCanvas: React.FC<GridCanvasProps> = ({
  tHours,
  flows,
  typeVisibility,
  onHover,
  onClickNode,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const iconImagesRef = useRef<Partial<Record<NodeType, HTMLImageElement>>>({});

  const zoomRef = useRef(1);
  const offsetRef = useRef({ x: 0, y: 0 });
  const isDraggingRef = useRef(false);
  const lastMouseRef = useRef<{ x: number; y: number } | null>(null);

  // load icons once
  useEffect(() => {
    const icons: Partial<Record<NodeType, HTMLImageElement>> = {};
    (Object.keys(ICON_PATHS) as NodeType[]).forEach((type) => {
      const src = ICON_PATHS[type];
      if (!src) return;
      const img = new Image();
      img.src = src;
      icons[type] = img;
    });
    iconImagesRef.current = icons;
  }, []);

  // drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpi = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 800;
    const cssHeight = canvas.clientHeight || 600;
    canvas.width = cssWidth * dpi;
    canvas.height = cssHeight * dpi;
    ctx.setTransform(dpi, 0, 0, dpi, 0, 0);

    const width = cssWidth;
    const height = cssHeight;

    const zoom = zoomRef.current;
    const offset = offsetRef.current;

    const worldWidth = 26;
    const baseScale = Math.min(width, height) / worldWidth;
    const scale = baseScale * zoom;

    const worldToCanvas = (x: number, y: number) => {
      const cx = width / 2 + x * scale + offset.x;
      const cy = height / 2 - y * scale + offset.y;
      return { cx, cy };
    };

    const nodeById: Record<string, GridNode> = {};
    NODES.forEach((n) => (nodeById[n.id] = n));

    // background
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#050814";
    ctx.fillRect(0, 0, width, height);

    // lines
    for (const line of LINES) {
      const f = flows[line.id];
      if (!f) continue;

      const fromNode = nodeById[line.from];
      const toNode = nodeById[line.to];

      const p1 = worldToCanvas(fromNode.x, fromNode.y);
      const p2 = worldToCanvas(toNode.x, toNode.y);

      let baseColor = "#27ae60";
      if (f.color === "yellow") baseColor = "#f1c40f";
      if (f.color === "red") baseColor = "#e74c3c";

      const dx = p2.cx - p1.cx;
      const dy = p2.cy - p1.cy;
      const coreWidth = 2 + 4 * Math.min(f.loading, 1.5);
      const glowWidth = coreWidth + 6;

      // glow
      ctx.save();
      ctx.lineCap = "round";
      ctx.strokeStyle =
        f.color === "green"
          ? "rgba(39,174,96,0.25)"
          : f.color === "yellow"
          ? "rgba(241,196,15,0.25)"
          : "rgba(231,76,60,0.25)";
      ctx.lineWidth = glowWidth;
      ctx.beginPath();
      ctx.moveTo(p1.cx, p1.cy);
      ctx.lineTo(p2.cx, p2.cy);
      ctx.stroke();
      ctx.restore();

      // core
      ctx.save();
      ctx.strokeStyle = baseColor;
      ctx.lineWidth = coreWidth;
      ctx.beginPath();
      ctx.moveTo(p1.cx, p1.cy);
      ctx.lineTo(p2.cx, p2.cy);
      ctx.stroke();
      ctx.restore();

      // moving blobs along the line, using tHours
      const elapsedSeconds = tHours * 3600;
      const cycleSeconds = 3;
      const phase = (elapsedSeconds / cycleSeconds) % 1;
      const direction = f.direction >= 0 ? 1 : -1;
      const numBlobs = 4;

      for (let i = 0; i < numBlobs; i++) {
        let t = (phase + (i / numBlobs) * 0.5) % 1;
        if (direction < 0) t = 1 - t;

        const bx = p1.cx + t * dx;
        const by = p1.cy + t * dy;

        const radius = coreWidth * 1.1;
        const gradient = ctx.createRadialGradient(
          bx,
          by,
          0,
          bx,
          by,
          radius * 2,
        );
        gradient.addColorStop(
          0,
          f.color === "green"
            ? "rgba(46,204,113,1.0)"
            : f.color === "yellow"
            ? "rgba(241,196,15,1.0)"
            : "rgba(231,76,60,1.0)",
        );
        gradient.addColorStop(1, "rgba(0,0,0,0)");

        ctx.save();
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(bx, by, radius * 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }

    // nodes (respect legend visibility)
    const icons = iconImagesRef.current;
    for (const node of NODES) {
      if (!typeVisibility[node.type]) continue;

      const { cx, cy } = worldToCanvas(node.x, node.y);
      const icon = icons[node.type];

      if (icon && icon.complete) {
        const size = 52;
        ctx.save();
        ctx.globalAlpha = 0.9;
        ctx.drawImage(icon, cx - size / 2, cy - size / 2, size, size);
        ctx.restore();
      } else {
        const color = NODE_COLOR[node.type];
        const radius = 10;
        ctx.save();
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#111";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
      }
    }
  }, [tHours, flows, typeVisibility]);

  // -------------- mouse / wheel handlers ---------------

  const findNearestNode = (mouseX: number, mouseY: number): GridNode | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const width = canvas.clientWidth || 800;
    const height = canvas.clientHeight || 600;
    const zoom = zoomRef.current;
    const offset = offsetRef.current;

    const worldWidth = 26;
    const baseScale = Math.min(width, height) / worldWidth;
    const scale = baseScale * zoom;

    let bestNode: GridNode | null = null;
    let bestDist2 = Infinity;
    const threshold = 26;

    for (const node of NODES) {
      const cx = width / 2 + node.x * scale + offset.x;
      const cy = height / 2 - node.y * scale + offset.y;
      const dx = mouseX - cx;
      const dy = mouseY - cy;
      const d2 = dx * dx + dy * dy;
      if (d2 < threshold * threshold && d2 < bestDist2) {
        bestDist2 = d2;
        bestNode = node;
      }
    }
    return bestNode;
  };

  const handleWheel: React.WheelEventHandler<HTMLCanvasElement> = (e) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const oldZoom = zoomRef.current;
    const newZoom = Math.min(Math.max(oldZoom * zoomFactor, 0.5), 4);

    const zoomRatio = newZoom / oldZoom;
    offsetRef.current = {
      x: mouseX - zoomRatio * (mouseX - offsetRef.current.x),
      y: mouseY - zoomRatio * (mouseY - offsetRef.current.y),
    };

    zoomRef.current = newZoom;
  };

  const handleMouseDown: React.MouseEventHandler<HTMLCanvasElement> = (e) => {
    isDraggingRef.current = true;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp: React.MouseEventHandler<HTMLCanvasElement> = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (lastMouseRef.current) {
      const dx = e.clientX - lastMouseRef.current.x;
      const dy = e.clientY - lastMouseRef.current.y;
      const moved2 = dx * dx + dy * dy;

      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      if (moved2 < 4 && onClickNode) {
        const node = findNearestNode(mouseX, mouseY);
        onClickNode({ node: node ?? null, x: mouseX, y: mouseY });
      }
    }

    isDraggingRef.current = false;
    lastMouseRef.current = null;
  };

  const handleMouseLeave: React.MouseEventHandler<HTMLCanvasElement> = () => {
    isDraggingRef.current = false;
    lastMouseRef.current = null;
    if (onHover) onHover({ node: null, x: 0, y: 0 });
  };

  const handleMouseMove: React.MouseEventHandler<HTMLCanvasElement> = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isDraggingRef.current && lastMouseRef.current) {
      const dx = e.clientX - lastMouseRef.current.x;
      const dy = e.clientY - lastMouseRef.current.y;
      offsetRef.current = {
        x: offsetRef.current.x + dx,
        y: offsetRef.current.y + dy,
      };
      lastMouseRef.current = { x: e.clientX, y: e.clientY };
    }

    if (onHover) {
      const node = findNearestNode(mouseX, mouseY);
      onHover({ node: node ?? null, x: mouseX, y: mouseY });
    }
  };

  return (
    <canvas
      ref={canvasRef}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onMouseMove={handleMouseMove}
      style={{
        width: "100%",
        height: "600px",
        borderRadius: "12px",
        background: "#050814",
        cursor: "grab",
      }}
    />
  );
};

export default GridCanvas;