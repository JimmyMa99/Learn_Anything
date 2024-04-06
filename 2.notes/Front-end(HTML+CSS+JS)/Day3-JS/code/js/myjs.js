import * as G2Plot from '@antv/g2plot'
const container = document.getElementById('app');
const data = [
  {
    "x": "2019-01",
    "y": 758
  },
  {
    "x": "2019-02",
    "y": 790
  },
  {
    "x": "2019-03",
    "y": 880
  },
  {
    "x": "2019-04",
    "y": 484
  },
  {
    "x": "2019-05",
    "y": 310
  },
  {
    "x": "2019-06",
    "y": 831
  }
];
const config = {
  "title": {
    "visible": true,
    "text": "单折线图"
  },
  "description": {
    "visible": true,
    "text": "一个简单的单折线图"
  },
  "legend": {
    "flipPage": false
  },
  "forceFit": false,
  "width": 560,
  "height": 376,
  "xField": "x",
  "yField": "y",
  "color": [
    "#5B8FF9"
  ]
}
const plot = new G2Plot.Line(container, {
  data,
  ...config,
});
plot.render();