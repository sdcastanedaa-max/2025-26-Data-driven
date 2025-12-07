// frontend/src/gridData.ts

// --- Types -------------------------------------------------------

export type NodeType =
  | "substation"
  | "wind"
  | "pv"
  | "fossil"
  | "nuclear"
  | "bess"
  | "load_res"
  | "load_ind";

export interface GridNode {
  id: string;
  name: string;
  type: NodeType;
  x: number;
  y: number;
  capacity_MW: number;
}

export interface GridLine {
  id: string;
  from: string;
  to: string;
  rating_MW: number;
  level: "T" | "D";
}

// -----------------------------------------------------------------
// Coordinate system: roughly -12..+12 in both x and y
// Arranged to loosely resemble Spain’s silhouette:
//   - x: west (-) to east (+)
//   - y: south (-) to north (+)
// -----------------------------------------------------------------

export const NODES: GridNode[] = [
  // --- Substations (6) -------------------------------------------
  {
    id: "S1",
    name: "Galicia Substation",
    type: "substation",
    x: -9,
    y: 5,
    capacity_MW: 6000,
  },
  {
    id: "S2",
    name: "Cantabria Substation",
    type: "substation",
    x: -3,
    y: 7,
    capacity_MW: 6000,
  },
  {
    id: "S3",
    name: "Catalonia Substation",
    type: "substation",
    x: 7,
    y: 6,
    capacity_MW: 6500,
  },
  {
    id: "S4",
    name: "Madrid Substation",
    type: "substation",
    x: 0,
    y: 3,
    capacity_MW: 8000,
  },
  {
    id: "S5",
    name: "Andalusia Substation",
    type: "substation",
    x: -4,
    y: -5,
    capacity_MW: 7000,
  },
  {
    id: "S6",
    name: "Cáceres Substation",
    type: "substation",
    x: -7,
    y: 0,
    capacity_MW: 4500,
  },

  // --- Wind farms (north & inland) -------------------------------
  {
    id: "W1",
    name: "Galicia Wind",
    type: "wind",
    x: -10,
    y: 7,
    capacity_MW: 700,
  },
  {
    id: "W2",
    name: "Cantabrian Wind",
    type: "wind",
    x: -5,
    y: 9,
    capacity_MW: 450,
  },
  {
    id: "W3",
    name: "Ebro Valley Wind",
    type: "wind",
    x: 2,
    y: 7,
    capacity_MW: 280,
  },
  {
    id: "W4",
    name: "Castilla Wind",
    type: "wind",
    x: -0.5,
    y: 4.5,
    capacity_MW: 800,
  },
  {
    id: "W5",
    name: "Catalan Wind",
    type: "wind",
    x: 8,
    y: 7.5,
    capacity_MW: 1000,
  },

  // --- PV farms (central & south) -------------------------------
  {
    id: "P1",
    name: "Extremadura Solar",
    type: "pv",
    x: -8,
    y: -1,
    capacity_MW: 150,
  },
  {
    id: "P2",
    name: "La Mancha Solar",
    type: "pv",
    x: 0,
    y: 0.5,
    capacity_MW: 550,
  },
  {
    id: "P3",
    name: "Valencia Solar",
    type: "pv",
    x: 7,
    y: 1,
    capacity_MW: 450,
  },
  {
    id: "P4",
    name: "Seville Solar",
    type: "pv",
    x: -6,
    y: -4.5,
    capacity_MW: 250,
  },
  {
    id: "P5",
    name: "Granada Solar",
    type: "pv",
    x: -1,
    y: -4.5,
    capacity_MW: 390,
  },
  {
    id: "P6",
    name: "Murcia Solar",
    type: "pv",
    x: 6,
    y: -4,
    capacity_MW: 280,
  },
  {
    id: "P7",
    name: "Central Plateau Solar",
    type: "pv",
    x: -2,
    y: 1.5,
    capacity_MW: 340,
  },
  {
    id: "P8",
    name: "Ebro Solar",
    type: "pv",
    x: 3.5,
    y: 4.5,
    capacity_MW: 450,
  },

  // --- Fossil plants (4) ----------------------------------------
  {
    id: "F1",
    name: "Atlantic CCGT",
    type: "fossil",
    x: -10,
    y: 3,
    capacity_MW: 700,
  },
  {
    id: "F2",
    name: "Asturias CCGT",
    type: "fossil",
    x: -4,
    y: 4.5,
    capacity_MW: 650,
  },
  {
    id: "F3",
    name: "Valencia CCGT",
    type: "fossil",
    x: 8,
    y: 1,
    capacity_MW: 700,
  },
  {
    id: "F4",
    name: "Guadalquivir CCGT",
    type: "fossil",
    x: -2,
    y: -7,
    capacity_MW: 750,
  },

  // --- Nuclear plants (2) ---------------------------------------
  {
    id: "N1",
    name: "Central Nuclear Norte",
    type: "nuclear",
    x: -1,
    y: 3.5,
    capacity_MW: 1000,
  },
  {
    id: "N2",
    name: "Central Nuclear Este",
    type: "nuclear",
    x: 3,
    y: 3.5,
    capacity_MW: 1000,
  },

  // --- BESS (2) -------------------------------------------------
  {
    id: "B1",
    name: "Madrid BESS",
    type: "bess",
    x: 0,
    y: 1.5,
    capacity_MW: 200,
  },
  {
    id: "B2",
    name: "Levante BESS",
    type: "bess",
    x: 4,
    y: -1.5,
    capacity_MW: 220,
  },

  // --- Loads: cities + industrial hubs --------------------------
  {
    id: "L1",
    name: "Madrid Metropolitan Load",
    type: "load_res",
    x: 1,
    y: 1.5,
    capacity_MW: 800,
  },
  {
    id: "L2",
    name: "Barcelona Metropolitan Load",
    type: "load_res",
    x: 8,
    y: 3.5,
    capacity_MW: 900,
  },
  {
    id: "L3",
    name: "Bilbao Metropolitan Load",
    type: "load_res",
    x: -3,
    y: 6,
    capacity_MW: 650,
  },
  {
    id: "L4",
    name: "Ebro Industrial Hub",
    type: "load_ind",
    x: 3,
    y: 2.5,
    capacity_MW: 900,
  },
  {
    id: "L5",
    name: "Andalusia Industrial Hub",
    type: "load_ind",
    x: -4,
    y: -3.5,
    capacity_MW: 950,
  },
];

  export const LINES: GridLine[] = [
  // --- Transmission backbone between substations -----------------
  { id: "L_S1_S2", from: "S1", to: "S2", rating_MW: 3500, level: "T" },
  { id: "L_S2_S4", from: "S2", to: "S4", rating_MW: 4000, level: "T" },
  { id: "L_S4_S3", from: "S4", to: "S3", rating_MW: 4200, level: "T" },
  { id: "L_S4_S5", from: "S4", to: "S5", rating_MW: 3800, level: "T" },
  { id: "L_S5_S6", from: "S5", to: "S6", rating_MW: 3600, level: "T" },
  { id: "L_S6_S1", from: "S6", to: "S1", rating_MW: 3600, level: "T" },
  { id: "L_S6_S4", from: "S6", to: "S4", rating_MW: 3400, level: "T" },

  // --- Wind → nearest substations --------------------------------
  { id: "L_W1_S1", from: "W1", to: "S1", rating_MW: 500, level: "T" },
  { id: "L_W2_S2", from: "W2", to: "S2", rating_MW: 450, level: "T" },
  { id: "L_W3_S4", from: "W3", to: "S4", rating_MW: 400, level: "T" },
  { id: "L_W4_S4", from: "W4", to: "S4", rating_MW: 800, level: "T" },
  { id: "L_W5_S3", from: "W5", to: "S3", rating_MW: 800, level: "T" },

  // --- PV → regional substations --------------------------------
  { id: "L_P1_S6", from: "P1", to: "S6", rating_MW: 250, level: "D" },
  { id: "L_P2_S4", from: "P2", to: "S4", rating_MW: 550, level: "D" },
  { id: "L_P3_S3", from: "P3", to: "S3", rating_MW: 450, level: "D" },
  { id: "L_P4_S5", from: "P4", to: "S5", rating_MW: 280, level: "D" },
  { id: "L_P5_S5", from: "P5", to: "S5", rating_MW: 280, level: "D" },
  { id: "L_P6_S3", from: "P6", to: "S3", rating_MW: 250, level: "D" },
  { id: "L_P7_S4", from: "P7", to: "S4", rating_MW: 260, level: "D" },
  { id: "L_P8_S4", from: "P8", to: "S4", rating_MW: 450, level: "D" },

  // --- Fossil → backbone -----------------------------------------
  { id: "L_F1_S1", from: "F1", to: "S1", rating_MW: 800, level: "T" },
  { id: "L_F2_S2", from: "F2", to: "S2", rating_MW: 750, level: "T" },
  { id: "L_F3_S3", from: "F3", to: "S3", rating_MW: 800, level: "T" },
  { id: "L_F4_S5", from: "F4", to: "S5", rating_MW: 850, level: "T" },

  // --- Nuclear → central substations -----------------------------
  { id: "L_N1_S4", from: "N1", to: "S4", rating_MW: 1200, level: "T" },
  { id: "L_N1_S2", from: "N1", to: "S2", rating_MW: 900, level: "T" },
  { id: "L_N2_S3", from: "N2", to: "S3", rating_MW: 1200, level: "T" },
  { id: "L_N2_S4", from: "N2", to: "S4", rating_MW: 900, level: "T" },

  // --- BESS connections -----------------------------------------
  { id: "L_B1_S4", from: "B1", to: "S4", rating_MW: 250, level: "D" },
  { id: "L_B2_S3", from: "B2", to: "S3", rating_MW: 250, level: "D" },

  // --- Loads: mostly fed from nearby substations -----------------
  { id: "L_L1_S4", from: "L1", to: "S4", rating_MW: 900, level: "D" },
  { id: "L_L2_S3", from: "L2", to: "S3", rating_MW: 1000, level: "D" },
  { id: "L_L3_S2", from: "L3", to: "S2", rating_MW: 800, level: "D" },
  { id: "L_L4_S4", from: "L4", to: "S4", rating_MW: 1000, level: "D" },
  { id: "L_L5_S5", from: "L5", to: "S5", rating_MW: 1100, level: "D" },

  // Extra tie from industrial Ebro hub to Catalonia substation
  { id: "L_L4_S3", from: "L4", to: "S3", rating_MW: 700, level: "D" },
];

