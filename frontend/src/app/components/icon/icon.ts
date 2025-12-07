import { Component, input } from '@angular/core';
import { IconName } from '../../types/icon';

@Component({
  selector: 'app-icon',
  imports: [],
  templateUrl: './icon.html',
  styleUrl: './icon.css'
})
export class Icon {
  name = input.required<IconName>();
  className = input<string>('w-6 h-6');
}
