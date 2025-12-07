import {
  AfterViewInit,
  Component,
  ElementRef,
  OnDestroy,
  ViewChild,
  ViewEncapsulation,
  computed,
  effect,
  input,
  output,
  signal
} from '@angular/core';

import cytoscape, { Core, NodeSingular } from 'cytoscape';
import { env } from '../../environments/env.dev';
import { SelectedNodesData } from '../../types/node-list';
import { ResultRes } from '../../types/result';
import { NodeList } from '../node-list/node-list';

type ColumnType = 'cnn' | 'img-txt' | 'full';

@Component({
  selector: 'app-graph-display',
  imports: [NodeList],
  templateUrl: './graph-display.html',
  styleUrl: './graph-display.css',
  encapsulation: ViewEncapsulation.None
})
export class GraphDisplay implements AfterViewInit, OnDestroy {
  result = input<ResultRes | null>(null);
  selectionData = input<SelectedNodesData | null>(null);
  columnType = input.required<ColumnType>();

  selectionChanged = output<SelectedNodesData>();

  @ViewChild('cyContainer', { static: false })
  cyContainer!: ElementRef<HTMLDivElement>;

  private cy: Core | null = null;
  private activeTooltipNode: NodeSingular | null = null;
  private audioElements = new Map<string, HTMLAudioElement>();

  tooltipInfo = signal({
    visible: false,
    x: 0,
    y: 0,
    word: '',
    sentence: ''
  });

  missingAudio = computed(() => this.columnType() === 'img-txt');
  weights = computed(() => {
    const res = this.result();
    if (!res) return null;

    const type = this.columnType();
    if (type === 'img-txt') return res.imgTxtWeights;
    if (type === 'full') return res.fullWeights;
    return null;
  });

  get cyVal() {
    return this.cy;
  }

  constructor() {
    effect(() => {
      const r = this.result();
      const s = this.selectionData();

      if (this.cy && r && s) {
        this.updateGraph();
      }
    });
  }

  ngAfterViewInit() {
    if (this.cyContainer) {
      this.initializeCytoscape();
    }
  }

  ngOnDestroy() {
    this.cy?.destroy();
    this.audioElements.forEach(a => a.pause());
    this.audioElements.clear();
  }

  onSelectionChanged(data: SelectedNodesData) {
    this.selectionChanged.emit(data);
  }

  // -------------------------------------------------
  // INIT CYTOSCAPE
  // -------------------------------------------------
  private initializeCytoscape() {
    if (!this.cyContainer) return;

    this.cy = cytoscape({
      container: this.cyContainer.nativeElement,
      wheelSensitivity: 0.2,
      style: this.getStyles(),
      layout: { name: 'grid' },
      boxSelectionEnabled: false
    });

    this.setupEventListeners();

    if (this.result() && this.selectionData()) {
      this.updateGraph();
    }
  }

  // -------------------------------------------------
  // STYLES
  // -------------------------------------------------
  private getStyles(): any[] {
    return [
      {
        selector: 'node',
        style: {
          'width': 60,
          'height': 60,
          'background-color': '#ffffff',
          'border-width': 0,
          'overlay-opacity': 0,
        }
      },
      {
        selector: 'node[type="image"]',
        style: {
          'shape': 'rectangle',
          'background-image': 'data(imageUrl)',
          'background-fit': 'cover',
          'background-clip': 'node',
          'background-opacity': 1,
          'border-width': 2,
          'border-color': '#10b981',
          'border-style': 'solid'
        }
      },
      {
        selector: 'node[type="text"]',
        style: {
          'label': 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '12px',
          'font-weight': 'bold',
          'color': '#ffffff',
          'background-color': '#22d3ee',
          'border-color': '#06b6d4',
          'border-width': 2,
          'text-wrap': 'wrap',
          'text-max-width': 80
        }
      },
      {
        selector: 'node[type="audio"]',
        style: {
          'label': '▶',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '24px',
          'color': '#ffffff',
          'background-color': '#3b82f6',
          'border-color': '#2563eb',
          'border-width': 2,
          'shape': 'round-rectangle'
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 'data(weight)',
          'line-color': '#94a3b8',
          'opacity': 0.8,
          'curve-style': 'bezier',
          'label': 'data(label)',
          'font-size': 10,
          'color': '#334155',
          'text-background-color': '#ffffff',
          'text-background-opacity': 0.8,
          'text-rotation': 'autorotate'
        }
      },
      {
        selector: 'node:selected',
        style: {
          'border-width': 4,
          'border-color': '#3b82f6',
          'border-style': 'solid',
          'overlay-opacity': 0
        }
      },
      {
        selector: 'node[type="text"]:selected',
        style: {
          'background-color': '#06b6d4',
        }
      },
      {
        selector: 'node[type="audio"]:selected',
        style: {
          'background-color': '#2563eb',
        }
      }
    ];
  }

  // -------------------------------------------------
  // EVENTS
  // -------------------------------------------------
  private setupEventListeners() {
    if (!this.cy) return;

    this.cy.on('tap', 'node[type="audio"]', (evt) => {
      const node = evt.target;
      const audioUrl = node.data('audioUrl');
      if (audioUrl) this.playAudio(node.id(), audioUrl);
    });

    this.cy.on('mouseover', 'node[type="text"]', evt => {
      const node = evt.target;
      this.activeTooltipNode = node;
      this.updateTooltipState(node);
    });

    this.cy.on('mouseout', 'node[type="text"]', () => {
      this.activeTooltipNode = null;
      this.tooltipInfo.update(t => ({ ...t, visible: false }));
    });

    this.cy.on('position pan zoom', () => {
      if (this.activeTooltipNode) {
        this.updateTooltipState(this.activeTooltipNode);
      }
    });
  }

  private updateTooltipState(node: NodeSingular) {
    const pos = node.renderedPosition();
    this.tooltipInfo.set({
      visible: true,
      x: pos.x,
      y: pos.y,
      word: node.data('word'),
      sentence: node.data('sentence')
    });
  }

  private playAudio(id: string, url: string) {
    this.audioElements.forEach(a => a.pause());

    let audio = this.audioElements.get(id);
    if (!audio) {
      audio = new Audio(url);
      this.audioElements.set(id, audio);
    }
    audio.currentTime = 0;
    audio.play().catch(console.error);
  }

  // -------------------------------------------------
  // GRAPH UPDATE
  // -------------------------------------------------
  private updateGraph() {
    if (!this.cy) return;

    const result = this.result();
    const sel = this.selectionData();
    const w = this.weights();

    if (!result || !sel || !w) return;

    this.cy.elements().remove();
    this.audioElements.clear();

    const elements = this.buildElements(result, sel);
    const edges = this.calculateEdges(elements, w, sel);

    this.cy.add([...elements, ...edges]);
    this.runLayout();
  }

  private runLayout() {
    if (!this.cy) return;
    const layout = this.cy.layout({
      name: 'cose',
      animate: true,
      animationDuration: 1000,
      randomize: false,
      nodeRepulsion: () => 8000,
      idealEdgeLength: () => 100,
      edgeElasticity: () => 100,
      gravity: 80,
      numIter: 1000,
      fit: true,
      padding: 50
    } as any);
    layout.run();
  }

  // -------------------------------------------------
  // ELEMENT BUILDERS
  // -------------------------------------------------
  private buildElements(result: ResultRes, sel: SelectedNodesData) {
    const out: any[] = [];
    let idx = 0;
    const skipAudio = this.missingAudio();

    // IMAGES
    sel.imageIndices.forEach(i => {
      if (result.imageUrls?.[i]) {
        out.push({
          data: {
            id: `img-${i}`,
            type: 'image',
            imageUrl: `${env.apiUrl}/${result.imageUrls[i]}`,
            matrixIndex: idx++
          }
        });
      }
    });

    // TEXTS
    sel.textIndices.forEach(i => {
      if (result.processedTexts?.[i]) {
        const item = result.processedTexts[i];
        const display = item.word.length > 10 ? item.word.slice(0, 10) + '…' : item.word;

        out.push({
          data: {
            id: `text-${i}`,
            type: 'text',
            word: item.word,
            sentence: item.sentence,
            label: display,
            matrixIndex: idx++
          }
        });
      }
    });

    // AUDIO (skip if img-txt mode)
    if (!skipAudio) {
      sel.audioIndices.forEach(i => {
        if (result.audioUrls?.[i]) {
          out.push({
            data: {
              id: `audio-${i}`,
              type: 'audio',
              audioUrl: `${env.apiUrl}/${result.audioUrls[i]}`,
              matrixIndex: idx++
            }
          });
        }
      });
    }

    return out;
  }

  private calculateEdges(nodes: any[], weights: number[][], sel: SelectedNodesData) {
    if (!weights) return [];

    const all: any[] = [];

    for (let a = 0; a < nodes.length; a++) {
      for (let b = a + 1; b < nodes.length; b++) {
        const ai = nodes[a].data.matrixIndex;
        const bi = nodes[b].data.matrixIndex;

        if (ai < weights.length && bi < weights[ai].length) {
          const w = weights[ai][bi];
          if (w > 0) {
            all.push({
              source: nodes[a].data.id,
              target: nodes[b].data.id,
              weight: Number(w.toFixed(4))
            });
          }
        }
      }
    }

    all.sort((a, b) => b.weight - a.weight);
    const keep = Math.ceil(all.length * (sel.threshold / 100));
    const selected = all.slice(0, keep);

    return selected.map((e, i) => ({
      data: {
        id: `edge-${i}`,
        source: e.source,
        target: e.target,
        weight: Math.max(1, Math.min(e.weight * 5, 10)),
        label: e.weight.toString()
      }
    }));
  }
}
