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
  cnnLabelIdx: number;
  cnnLabelName: string;
  cnnProb: number;
  imgTxtLabelIdx: number;
  imgTxtLabelName: string;
  imgTxtProb: number;
  imgTxtWeights: number[][];
  fullLabelIdx: string;
  fullLabelName: string;
  fullProb: number;
  fullWeights: number[][];
}
