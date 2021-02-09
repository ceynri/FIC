import Vue from 'vue';

import { OverlayScrollbarsPlugin, OverlayScrollbarsComponent } from 'overlayscrollbars-vue';
import OverlayScrollbars from 'overlayscrollbars';

import 'overlayscrollbars/css/OverlayScrollbars.css';

Vue.use(OverlayScrollbarsPlugin);

Vue.component('overlay-scrollbars', OverlayScrollbarsComponent);

OverlayScrollbars(document.body, {
  nativeScrollbarsOverlaid: {
    initialize: false,
  },
});
