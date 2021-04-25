import Vue from 'vue';

import { OverlayScrollbarsPlugin, OverlayScrollbarsComponent } from 'overlayscrollbars-vue';

import 'overlayscrollbars/css/OverlayScrollbars.css';

Vue.use(OverlayScrollbarsPlugin);

Vue.component('overlay-scrollbars', OverlayScrollbarsComponent);
