import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HompageVideoComponent } from './hompage-video.component';

describe('HompageVideoComponent', () => {
  let component: HompageVideoComponent;
  let fixture: ComponentFixture<HompageVideoComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ HompageVideoComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(HompageVideoComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
