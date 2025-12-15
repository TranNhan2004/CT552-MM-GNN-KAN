export type PredictionType = "img" | "img-txt" | "full"

export interface PredictionResult {
  labelName: string;
  prob: number;
  weights?: number[][];
}

export interface UploadRes {
  id: number;
}

interface Word {
  word: string;
  sentence: string;
}

export interface ResultRes {
  id: number;
  text: string;
  processedTexts: Word[];
  imageUrls: string[];
  audioUrls: string[];
  imgLabelIdx: number;
  imgLabelName: string;
  imgProb: number;
  imgTxtLabelIdx: number;
  imgTxtLabelName: string;
  imgTxtProb: number;
  imgTxtWeights: number[][];
  fullLabelIdx: string;
  fullLabelName: string;
  fullProb: number;
  fullWeights: number[][];
}
