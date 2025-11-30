import { Component, Input } from '@angular/core';
import { IconName } from '../../types/icon';

@Component({
  selector: 'app-icon',
  imports: [],
  templateUrl: './icon.html',
  styleUrl: './icon.css'
})
export class Icon {
  @Input({ required: true }) name!: IconName;
  @Input() className: string = 'w-6 h-6';
}
