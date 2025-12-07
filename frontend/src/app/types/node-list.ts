export type EdgeThreshold = 100 | 80 | 60 | 40 | 20;

export interface SelectedNodesData {
  imageIndices: number[];
  textIndices: number[];
  audioIndices: number[];
  threshold: EdgeThreshold;
}
