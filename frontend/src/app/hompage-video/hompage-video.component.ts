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
        this.showResults = true;
        this.imageSrc = reader.result;
      };

      reader.readAsDataURL(file);
    }
  }

  deepfry(event) {
    this.filterValue = event.target.value;
    console.log(this.filterValue);

    if (this.filterValue == 0) this.fliter = 'contrast(200%)';
    console.log(this.filterValue)
    this.fliter = `contrast(${(this.filterValue + 100) * 2}%) saturate(${
      (this.filterValue + 100) * 2
    }%) brightness(${((this.filterValue / 100 + 1) * 1.5).toString()})`;

    console.log(this.fliter);

    this.changeRef.detectChanges();
  }
}
