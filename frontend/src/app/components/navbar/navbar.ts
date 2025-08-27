import { Component, ElementRef, HostListener } from '@angular/core';

@Component({
  selector: 'app-navbar',
  imports: [],
  templateUrl: './navbar.html',
  styleUrl: './navbar.css'
})
export class Navbar {
  dropdownOpen = false;
  models = ['GPT-4', 'GPT-3.5', 'Custom-LLM', 'Other'];
  selectedModel = this.models[0];

  constructor(private eRef: ElementRef) { }

  toggleDropdown() {
    this.dropdownOpen = !this.dropdownOpen;
  }

  selectModel(model: string) {
    this.selectedModel = model;
    this.dropdownOpen = false;
  }

  @HostListener('document:click', ['$event'])
  clickOutside(event: Event) {
    if (this.dropdownOpen && !this.eRef.nativeElement.contains(event.target)) {
      this.dropdownOpen = false;
    }
  }
}
