import { ChangeDetectorRef, Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-hompage-video',
  templateUrl: './hompage-video.component.html',
  styleUrls: ['./hompage-video.component.scss'],
})
export class HompageVideoComponent implements OnInit {
  imageSrc: String;
  showResults: boolean;
  filterValue: number = 0;
  fliter: string = '';
  changeRef: ChangeDetectorRef;
  constructor(changeRef: ChangeDetectorRef) {
    this.changeRef = changeRef;
  }
  ngOnInit(): void {}
  handleFileInput(event) {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];

      const reader = new FileReader();

      reader.onload = (e) => {
        setTimeout(() => (this.showResults = true), 3000);
        this.imageSrc = reader.result;
      };

      reader.readAsDataURL(file);
    }
  }

  deepfry(event) {
    this.filterValue = event.target.value;

    if (this.filterValue == 0) this.fliter = '';
    else
      this.fliter = `contrast(${this.filterValue * 2 + 100}%) saturate(${
        this.filterValue * 2 + 100
      }%) brightness(${(this.filterValue / 200 + 1).toString()})`;
    this.changeRef.detectChanges();
  }
}
