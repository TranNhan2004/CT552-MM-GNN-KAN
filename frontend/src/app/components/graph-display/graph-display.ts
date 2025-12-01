import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, ElementRef, Input, OnChanges, OnDestroy, SimpleChanges, ViewChild, ViewEncapsulation } from '@angular/core';
import cytoscape, { Core, NodeSingular } from 'cytoscape';
import { env } from '../../environments/env.dev';
import { SelectedNodesData } from '../../types/node-list';
import { ResultRes } from '../../types/result';

@Component({
  selector: 'app-graph-display',
  imports: [CommonModule],
  templateUrl: './graph-display.html',
  styleUrl: './graph-display.css',
  encapsulation: ViewEncapsulation.None
})
export class GraphDisplay implements OnChanges, AfterViewInit, OnDestroy {
  @Input() result: ResultRes | null = null;
  @Input() selectionData: SelectedNodesData | null = null;
  @ViewChild('cyContainer', { static: false }) cyContainer!: ElementRef<HTMLDivElement>;

  private cy: Core | null = null;
  private audioElements: Map<string, HTMLAudioElement> = new Map();

  get cyVal() {
    return this.cy;
  }

  tooltipInfo = {
    visible: false,
    x: 0,
    y: 0,
    word: '',
    sentence: ''
  };

  private activeTooltipNode: NodeSingular | null = null;

  ngAfterViewInit() {
    if (this.cyContainer) {
      this.initializeCytoscape();
    }
  }

  ngOnChanges(changes: SimpleChanges) {
    if ((changes['result'] || changes['selectionData']) && this.cy) {
      this.updateGraph();
    }
  }

  ngOnDestroy() {
    if (this.cy) {
      this.cy.destroy();
    }
    this.audioElements.forEach(audio => audio.pause());
    this.audioElements.clear();
  }

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
    this.updateGraph();
  }

  private getStyles(): any[] {
    return [
      // 1. STYLE CƠ BẢN CHO TẤT CẢ NODE
      {
        selector: 'node',
        style: {
          'width': 60,
          'height': 60,
          'background-color': '#ffffff',
          'border-width': 0,
          // Quan trọng: Tắt overlay mặc định để không bị tối màu khi click
          'overlay-opacity': 0,
        }
      },

      // 2. IMAGE NODE (Định nghĩa ảnh ở đây duy nhất 1 lần)
      {
        selector: 'node[type="image"]',
        style: {
          'shape': 'rectangle', // Ảnh thì nên dùng hình chữ nhật hoặc vuông
          'background-image': 'data(imageUrl)',
          'background-fit': 'cover', // Cover để ảnh luôn full node
          'background-clip': 'node', // Cắt ảnh theo hình dáng node
          'background-opacity': 1,
          'border-width': 2,
          'border-color': '#10b981', // Border xanh lá mặc định
          'border-style': 'solid'
        }
      },

      // 3. TEXT NODE
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

      // 4. AUDIO NODE
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

      // 5. EDGE STYLE
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

      // -----------------------------------------------------------
      // KHU VỰC SỬA LỖI (SELECTED STATE)
      // -----------------------------------------------------------

      // Khi chọn bất kỳ node nào: Chỉ thêm border màu Blue (#3b82f6)
      // KHÔNG khai báo lại background-image ở đây
      {
        selector: 'node:selected',
        style: {
          'border-width': 4,
          'border-color': '#3b82f6', // Màu blue bạn muốn
          'border-style': 'solid',

          // Đảm bảo không có lớp phủ xám đè lên ảnh
          'overlay-opacity': 0
        }
      },

      // Nếu bạn muốn Text Node khi chọn có màu nền đậm hơn chút (optional)
      {
        selector: 'node[type="text"]:selected',
        style: {
          'background-color': '#06b6d4', // Cyan đậm hơn chút
        }
      },

      // Audio node khi chọn
      {
        selector: 'node[type="audio"]:selected',
        style: {
          'background-color': '#2563eb', // Blue đậm hơn chút
        }
      }
    ];
  }

  private setupEventListeners() {
    if (!this.cy) return;

    this.cy.on('tap', 'node[type="audio"]', (event) => {
      const node = event.target;
      const audioUrl = node.data('audioUrl');
      if (audioUrl) this.playAudio(node.id(), audioUrl);
    });

    this.cy.on('mouseover', 'node[type="text"]', (event) => {
      const node = event.target;
      this.activeTooltipNode = node;
      this.updateTooltipState(node);
    });

    this.cy.on('mouseout', 'node[type="text"]', () => {
      this.activeTooltipNode = null;
      this.hideTooltip();
    });

    this.cy.on('position pan zoom', () => {
      if (this.activeTooltipNode) {
        this.updateTooltipState(this.activeTooltipNode);
      }
    });
  }

  private updateTooltipState(node: any) {
    const renderedPos = node.renderedPosition();
    this.tooltipInfo = {
      visible: true,
      x: renderedPos.x,
      y: renderedPos.y,
      word: node.data('word'),
      sentence: node.data('sentence')
    };
  }

  private hideTooltip() {
    this.tooltipInfo = { ...this.tooltipInfo, visible: false };
  }

  private playAudio(nodeId: string, url: string) {
    this.audioElements.forEach(audio => audio.pause());
    let audio = this.audioElements.get(nodeId);
    if (!audio) {
      audio = new Audio(url);
      this.audioElements.set(nodeId, audio);
    }
    audio.currentTime = 0;
    audio.play().catch(e => console.error("Audio play failed", e));
  }

  private updateGraph() {
    if (!this.cy || !this.result || !this.selectionData) return;

    this.cy.elements().remove();
    this.audioElements.clear();

    const elements: any[] = [];
    let nodeIndex = 0;

    // Add Images
    this.selectionData.imageIndices.forEach(idx => {
      if (this.result!.imageUrls && idx < this.result!.imageUrls.length) {
        elements.push({
          data: {
            id: `img-${idx}`,
            type: 'image',
            imageUrl: `${env.apiUrl}/${this.result!.imageUrls[idx]}`,
            matrixIndex: nodeIndex
          }
        });
        nodeIndex++;
      }
    });

    // Add Texts
    this.selectionData.textIndices.forEach(idx => {
      if (this.result!.processedTexts && idx < this.result!.processedTexts.length) {
        const item = this.result!.processedTexts[idx];
        const maxLen = 10;
        let displayLabel = item.word.length > maxLen ? item.word.slice(0, maxLen) + '…' : item.word;

        elements.push({
          data: {
            id: `text-${idx}`,
            type: 'text',
            word: item.word,
            sentence: item.sentence,
            label: displayLabel,
            matrixIndex: nodeIndex
          }
        });
        nodeIndex++;
      }
    });

    // Add Audio
    this.selectionData.audioIndices.forEach(idx => {
      if (this.result!.audioUrls && idx < this.result!.audioUrls.length) {
        elements.push({
          data: {
            id: `audio-${idx}`,
            type: 'audio',
            audioUrl: `${env.apiUrl}/${this.result!.audioUrls[idx]}`,
            matrixIndex: nodeIndex
          }
        });
        nodeIndex++;
      }
    });

    // Add Edges
    const edges = this.calculateEdges(elements);
    elements.push(...edges);

    this.cy.add(elements);

    // Layout
    const layout = this.cy.layout({
      name: 'cose',
      animate: true,
      animationDuration: 1000,
      randomize: false,
      nodeRepulsion: (node: any) => 8000,
      idealEdgeLength: (edge: any) => 100,
      edgeElasticity: (edge: any) => 100,
      nestingFactor: 1.2,
      gravity: 80,
      numIter: 1000,
      initialTemp: 200,
      coolingFactor: 0.95,
      minTemp: 1.0,
      fit: true,
      padding: 50
    } as any);

    layout.run();
  }

  private calculateEdges(nodes: any[]): any[] {
    if (!this.result || !this.result.weights || !this.selectionData) return [];

    const edges: any[] = [];
    const allEdges: { source: string; target: string; weight: number }[] = [];

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const sourceIdx = nodes[i].data.matrixIndex;
        const targetIdx = nodes[j].data.matrixIndex;

        if (sourceIdx < this.result.weights.length &&
            targetIdx < this.result.weights[sourceIdx].length) {
          const weight = this.result.weights[sourceIdx][targetIdx];
          if (weight > 0) {
            allEdges.push({
              source: nodes[i].data.id,
              target: nodes[j].data.id,
              weight: Number(weight.toFixed(4))
            });
          }
        }
      }
    }

    allEdges.sort((a, b) => b.weight - a.weight);
    const topCount = Math.ceil(allEdges.length * (this.selectionData.threshold / 100));
    const selectedEdges = allEdges.slice(0, topCount);

    selectedEdges.forEach((edge, idx) => {
      const displayWidth = Math.max(1, Math.min(edge.weight * 5, 10));
      edges.push({
        data: {
          id: `edge-${idx}`,
          source: edge.source,
          target: edge.target,
          weight: displayWidth,
          label: edge.weight.toString()
        }
      });
    });

    return edges;
  }
}
